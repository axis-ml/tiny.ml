open Common
open Lang

module Ir = struct
  open Interp

  type var =
    { id : int
    ; aval : Aval.t
    }

  type eqn =
    { prim : Prim.t
    ; inputs : var list
    ; outputs : var list
    }

  type t =
    { params : var list
    ; captures : (var * value) list
    ; consts : (var * Ndarray.t) list
    ; eqns : eqn list
    ; outputs : var list
    ; num_vars : int
    }

  let avals xs = List.map (fun (x : var) -> x.aval) xs
  let ids xs = List.map (fun (x : var) -> x.id) xs

  let sub_graph_rules : (Prim.t -> (string * t) list) Prim.Table.t =
    Prim.Table.create "Ir.sub_graphs"
  ;;

  let register_sub_graphs p f = Prim.Table.add sub_graph_rules p f

  let sub_graphs p =
    match Prim.Table.find_opt sub_graph_rules p with
    | Some f -> f p
    | None -> []
  ;;

  exception Bad of string

  let fail fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt

  let rec check g =
    let in_range (x : var) = x.id >= 0 && x.id < g.num_vars in
    let define defined (x : var) =
      match in_range x, Env.mem x.id defined with
      | false, _ -> fail "%%%d is out of range (%d vars)" x.id g.num_vars
      | true, true -> fail "%%%d is defined twice" x.id
      | true, false -> Env.add x.id x.aval defined
    in
    let use defined (x : var) =
      match if in_range x then Env.find_opt x.id defined else None with
      | None -> fail "%%%d is used before it is defined" x.id
      | Some a when not (Aval.equal a x.aval) ->
        fail
          "%%%d is used as %s but defined as %s"
          x.id
          (Aval.to_string x.aval)
          (Aval.to_string a)
      | Some _ -> ()
    in
    let define_holding what defined ((x : var), a) =
      if Aval.equal a x.aval
      then define defined x
      else
        fail
          "%s %%%d is a %s holding a %s"
          what
          x.id
          (Aval.to_string x.aval)
          (Aval.to_string a)
    in
    let check_sub prim (name, sub) =
      try check sub with
      | Bad msg -> fail "%s.%s: %s" (Prim.name prim) name msg
    in
    let check_eqn defined (e : eqn) =
      List.iter (use defined) e.inputs;
      List.iter (check_sub e.prim) (sub_graphs e.prim);
      let expected =
        try Prim.abstract_eval e.prim (avals e.inputs) with
        | Error.Error (Type_error msg) -> fail "ill-typed eqn: %s" msg
      in
      if not (List.equal Aval.equal expected (avals e.outputs))
      then fail "%s: outputs do not have the types abstract_eval gives" (Prim.name e.prim);
      List.fold_left define defined e.outputs
    in
    let fold f xs env = List.fold_left f env xs in
    let captures = List.map (fun (c, v) -> c, aval v) g.captures in
    let consts = List.map (fun (c, a) -> c, Aval.of_ndarray a) g.consts in
    let defined =
      Env.empty
      |> fold define g.params
      |> fold (define_holding "capture") captures
      |> fold (define_holding "const") consts
      |> fold check_eqn g.eqns
    in
    List.iter (use defined) g.outputs
  ;;

  let invariant g =
    try check g with
    | Bad msg -> failwith @@ "Ir.invariant: " ^ msg
  ;;

  let trace_aux f in_avals =
    let ids = Graph.counter () in
    let fresh aval = { id = Graph.fresh ids; aval } in
    let eqns = Tape.create () in
    let free = Tape.create () in
    let seen v = Tape.find_map (fun (v', x) -> if v' == v then Some x else None) free in
    let add_free v =
      let x = fresh (aval v) in
      Tape.record free (v, x);
      x
    in
    let var_of stage v =
      match payload stage v with
      | Some x -> x
      | None ->
        (match seen v with
         | Some x -> x
         | None -> add_free v)
    in
    let process self prim args =
      let inputs = List.map (var_of self) args in
      let outputs = args |> List.map aval |> Prim.abstract_eval prim |> List.map fresh in
      Tape.record eqns { prim; inputs; outputs };
      List.map (fun x -> tracer self x.aval x) outputs
    in
    let stage = create ~name:"stage" process in
    let params = List.map fresh in_avals in
    let outs, aux =
      handle stage (fun () -> List.map (fun x -> tracer stage x.aval x) params |> f)
    in
    let outputs = List.map (var_of stage) outs in
    let consts, captures =
      Tape.to_list free
      |> List.partition_map (fun (v, x) ->
        match to_ndarray_opt v with
        | Some a -> Either.Left (x, a)
        | None -> Either.Right (x, v))
    in
    let g =
      { params
      ; captures
      ; consts
      ; eqns = Tape.to_list eqns
      ; outputs
      ; num_vars = Graph.count ids
      }
    in
    if !Debug.enabled then invariant g;
    g, aux
  ;;

  let trace f in_avals = trace_aux (fun xs -> f xs, ()) in_avals |> fst

  let make_tree ~input:in_tree ~output:out_tree f x =
    let leaves, s = Tree.flatten in_tree x in
    List.map aval leaves
    |> trace (fun tracers -> Tree.flat_fn in_tree out_tree f s tracers |> fst)
  ;;

  let make f x = make_tree ~input:Tree.value ~output:Tree.value f x

  let closure_convert g =
    ( { g with params = List.map fst g.captures @ g.params; captures = [] }
    , List.map snd g.captures )
  ;;

  let check_run_inputs g xs =
    if List.length xs <> List.length g.params
    then
      Error.type_error
        "Ir.run: expected %d input(s), got %d"
        (List.length g.params)
        (List.length xs);
    List.iter
      (fun (_, v) ->
         if not (alive v)
         then
           Error.escaped_tracer
             "Ir.run: the graph captured %s, whose transformation has returned; pass it \
              as an argument instead"
             (describe v))
      g.captures;
    List.combine g.params xs
    |> List.iteri (fun i ((p : var), x) ->
      if not (Aval.equal (aval x) p.aval)
      then
        Error.type_error
          "Ir.run: input %d is %s, expected %s"
          i
          (Aval.to_string (aval x))
          (Aval.to_string p.aval))
  ;;

  let read env (x : var) =
    match Env.find_opt x.id env with
    | Some v -> v
    | None -> Printf.ksprintf invalid_arg "Ir.run: %%%d is used before it is defined" x.id
  ;;

  let run_eqn env (e : eqn) =
    let outs = List.map (read env) e.inputs |> bind e.prim in
    if List.length outs <> List.length e.outputs
    then
      Printf.ksprintf
        invalid_arg
        "Ir.run: %s gave %d outputs, the eqn has %d"
        (Prim.name e.prim)
        (List.length outs)
        (List.length e.outputs);
    Env.add_all (ids e.outputs) outs env
  ;;

  let run g xs =
    check_run_inputs g xs;
    let capture_vars, captured = List.split g.captures in
    let const_vars, consts = List.split g.consts in
    let env =
      Env.empty
      |> Env.add_all (ids g.params) xs
      |> Env.add_all (ids capture_vars) captured
      |> Env.add_all (ids const_vars) (List.map of_ndarray consts)
    in
    let env = List.fold_left run_eqn env g.eqns in
    List.map (read env) g.outputs
  ;;

  let run1 g x =
    match run g [ x ] with
    | [ y ] -> y
    | ys -> Error.type_error "Ir.run1: expected 1 output, got %d" (List.length ys)
  ;;

  let producers g =
    let define defs (e : eqn) =
      List.fold_left (fun defs x -> Env.add x.id e defs) defs e.outputs
    in
    let defs = List.fold_left define Env.empty g.eqns in
    Array.init g.num_vars (fun id -> Env.find_opt id defs)
  ;;

  let users g =
    let add_user e = function
      | None -> Some [ e ]
      | Some es when List.memq e es -> Some es
      | Some es -> Some (e :: es)
    in
    let use uses (e : eqn) =
      List.fold_left (fun uses x -> Env.update x.id (add_user e) uses) uses e.inputs
    in
    let uses = List.rev g.eqns |> List.fold_left use Env.empty in
    Array.init g.num_vars (fun id -> Env.find_opt id uses |> Option.value ~default:[])
  ;;

  let live_mask g =
    Graph.live
      ~inputs:(fun (e : eqn) -> ids e.inputs)
      ~outputs:(fun (e : eqn) -> ids e.outputs)
      ~roots:(ids g.outputs)
      g.eqns
  ;;

  let dead g = Lists.reject (live_mask g) g.eqns
  let name x = Printf.sprintf "%%%d" x.id
  let typed x = name x ^ ":" ^ Aval.to_string x.aval
  let names xs = List.map name xs |> String.concat ", "
  let typed_names xs = List.map typed xs |> String.concat ", "

  let rec lines indent g =
    let line s = indent ^ s in
    let sub_lines (n, sub) = line ("  " ^ n ^ ":") :: lines (indent ^ "    ") sub in
    let eqn_lines (e : eqn) =
      let head =
        Printf.sprintf
          "%s = %s(%s)"
          (typed_names e.outputs)
          (Prim.name e.prim)
          (names e.inputs)
      in
      line head :: List.concat_map sub_lines (sub_graphs e.prim)
    in
    let capture (c, v) = line (typed c ^ " = capture " ^ describe v) in
    let const (c, a) = line (typed c ^ " = const " ^ Ndarray.to_string a) in
    [ line ("params: " ^ typed_names g.params) ]
    @ List.map capture g.captures
    @ List.map const g.consts
    @ List.concat_map eqn_lines g.eqns
    @ [ line ("outputs: " ^ names g.outputs) ]
  ;;

  let to_string g = lines "" g |> List.map (fun l -> l ^ "\n") |> String.concat ""
  let pp fmt g = Format.pp_print_string fmt (to_string g)

  let escape s =
    let escape_char = function
      | '"' -> "\\\""
      | '\\' -> "\\\\"
      | c -> String.make 1 c
    in
    String.to_seq s |> List.of_seq |> List.map escape_char |> String.concat ""
  ;;

  let rec dot_lines k clusters g =
    let sp = Printf.sprintf in
    let node s = sp "g%d_%s" k s in
    let var_node (x : var) = node (sp "v%d" x.id) in
    let edge src dst x = sp "  %s -> %s [label=\"%s\"];\n" src dst (name x) in
    let leaf shape label x =
      sp "  %s [shape=%s label=\"%s\"];\n" (var_node x) shape (escape label)
    in
    let leaves =
      List.map (fun x -> x, leaf "ellipse" (typed x) x) g.params
      @ List.map (fun (x, _) -> x, leaf "diamond" (typed x ^ " capture") x) g.captures
      @ List.map (fun (x, _) -> x, leaf "note" (typed x ^ " const") x) g.consts
    in
    let src =
      List.fold_left (fun src (x, _) -> Env.add x.id (var_node x) src) Env.empty leaves
    in
    let source src (x : var) = Option.value (Env.find_opt x.id src) ~default:"" in
    let eqn_lines (src, clusters) (i, (live, (e : eqn))) =
      let n = node (sp "e%d" i) in
      let box =
        sp
          "  %s [shape=box label=\"%s\"%s];\n"
          n
          (escape (Prim.name e.prim))
          (if live then "" else " style=dashed")
      in
      let ins = List.map (fun x -> edge (source src x) n x) e.inputs in
      let src = List.fold_left (fun src (x : var) -> Env.add x.id n src) src e.outputs in
      let cluster clusters (sub_name, sub) =
        let k' = clusters + 1 in
        let body, clusters = dot_lines k' k' sub in
        let header =
          sp
            "  subgraph cluster_%d {\n  label=\"%s.%s\"; style=dotted;\n"
            k'
            (escape (Prim.name e.prim))
            (escape sub_name)
        in
        let footer = [ "  }\n"; sp "  g%d_out -> %s [style=dotted];\n" k' n ] in
        clusters, (header :: body) @ footer
      in
      let clusters, subs = sub_graphs e.prim |> List.fold_left_map cluster clusters in
      (src, clusters), (box :: ins) @ List.concat subs
    in
    let (src, clusters), eqns =
      List.combine (live_mask g) g.eqns
      |> List.mapi (fun i x -> i, x)
      |> List.fold_left_map eqn_lines (src, clusters)
    in
    let out = node "out" in
    let out_lines =
      sp "  %s [shape=plaintext label=\"outputs\"];\n" out
      :: List.map (fun x -> edge (source src x) out x) g.outputs
    in
    List.map snd leaves @ List.concat eqns @ out_lines, clusters
  ;;

  let to_dot g =
    let body, _ = dot_lines 0 0 g in
    ("digraph ir {\n  node [fontname=\"monospace\"];\n" :: body) @ [ "}\n" ]
    |> String.concat ""
  ;;

  module Unchecked = struct
    let var ~id aval = { id; aval }
    let eqn prim ~inputs ~outputs = { prim; inputs; outputs }

    let graph ?(captures = []) ?(consts = []) ~params ~eqns ~outputs ~num_vars () =
      { params; captures; consts; eqns; outputs; num_vars }
    ;;
  end
end

module Util = struct
  open Interp

  let split = Lists.split_at
  let pick = Lists.pick
  let reject = Lists.reject
  let merge = Lists.merge
  let avals = List.map aval
  let avals_str l = List.map Aval.to_string l |> String.concat ", "
  let nonzero = List.map Autodiff.Maybe_zero.nonzero
  let instantiate = List.map Autodiff.Maybe_zero.instantiate
  let zero_like = List.map Autodiff.Maybe_zero.zero_like
  let zeros = List.map (fun (a : Aval.t) -> Ops.zeros a.shape)
  let is_nonzero = List.map (fun d -> not (Autodiff.Maybe_zero.is_zero d))

  let traced_aux f in_avals =
    match Ir.trace_aux f in_avals with
    | ({ Ir.captures = []; _ }, _) as r -> r
    | _ -> failwith "Staging: a rule's graph captured a tracer"
  ;;

  let traced f in_avals = traced_aux (fun xs -> f xs, ()) in_avals |> fst

  let empty =
    { Ir.params = []; captures = []; consts = []; eqns = []; outputs = []; num_vars = 0 }
  ;;

  let run_arrays g arrays = List.map of_ndarray arrays |> Ir.run g |> List.map to_ndarray
  let as_fn g xs = Ir.run g xs, ()

  let expect what ~graph ~operands =
    if not (List.equal Aval.equal graph operands)
    then
      Error.type_error
        "%s: operands are (%s), the graph takes (%s)"
        what
        (avals_str operands)
        (avals_str graph)
  ;;

  let fixed_values ops =
    List.filter_map
      (function
        | Autodiff.Linear.Fixed v -> Some v
        | Linear _ -> None)
      ops
  ;;

  let linear_avals ops =
    List.filter_map
      (function
        | Autodiff.Linear.Linear a -> Some a
        | Fixed _ -> None)
      ops
  ;;

  let is_linear = function
    | Autodiff.Linear.Linear _ -> true
    | Fixed _ -> false
  ;;

  let transposed (g : Ir.t) ops fixed cts =
    let fixed = List.map (fun v -> Autodiff.Linear.Fixed v) fixed in
    let ops = merge (List.map is_linear ops) (List.filter is_linear ops) fixed in
    let lin, () = Autodiff.Partial_eval.trace (as_fn g) ops in
    Autodiff.Transpose.transpose lin cts
  ;;
end

module Compile = struct
  open Interp
  open Util
  module Z = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Linear = Autodiff.Linear
  module Partial_eval = Autodiff.Partial_eval
  module Transpose = Autodiff.Transpose

  module type Backend = sig
    val name : string
    val compile : (value list -> value list * 'aux) -> value list -> value list * 'aux
  end

  type backend = (module Backend)

  type Prim.t +=
    | Call of
        { name : string
        ; graph : Ir.t
        }

  let call name graph xs = bind (Call { name; graph }) xs
  let wrap prefix name = prefix ^ "(" ^ name ^ ")"

  let call_abstract_eval p avals =
    match p with
    | Call { name; graph = { captures = []; _ } as g } ->
      expect name ~graph:(Ir.avals g.params) ~operands:avals;
      Ir.avals g.outputs
    | Call { name; _ } -> Error.type_error "%s: the graph captures tracers" name
    | _ -> Prim.arity_error p ~expected:0 avals
  ;;

  let call_impl p arrays =
    match p with
    | Call { graph; _ } -> run_arrays graph arrays
    | _ -> Prim.arity_error p ~expected:0 arrays
  ;;

  let call_jvp p xs dxs =
    match p with
    | Call { name; graph = g } ->
      let in_mask = is_nonzero dxs in
      let ts = instantiate (pick in_mask dxs) in
      let jvp_graph, out_mask =
        avals xs @ avals ts
        |> traced_aux (fun l ->
          let xs, ts = split (List.length xs) l in
          let dxs = merge in_mask (nonzero ts) (reject in_mask dxs) in
          let ys, dys, () = Jvp.jvp_flat (as_fn g) xs dxs in
          let out_mask = is_nonzero dys in
          ys @ instantiate (pick out_mask dys), out_mask)
      in
      let ys, dys =
        call (wrap "jvp" name) jvp_graph (xs @ ts) |> split (List.length g.outputs)
      in
      let zeros = reject out_mask ys |> zero_like in
      ys, merge out_mask (nonzero dys) zeros
    | _ -> Prim.arity_error p ~expected:0 xs
  ;;

  let run_linear (lin : Linear.t) ~known inputs =
    let read env = function
      | Linear.Var x ->
        (match Env.find_opt x.id env with
         | Some v -> v
         | None -> invalid_arg "Compile: a linear var is used before it is defined")
      | Known v -> known v
    in
    let ids = List.map (fun (x : Linear.var) -> x.id) in
    let run_eqn env (e : Linear.eqn) =
      let outs = List.map (read env) e.inputs |> bind e.prim in
      Env.add_all (ids e.outputs) outs env
    in
    let env = Env.add_all (ids lin.inputs) inputs Env.empty in
    List.fold_left run_eqn env lin.eqns |> read
  ;;

  let residuals (lin : Linear.t) =
    let known_tracer = function
      | Linear.Known v when Option.is_none (to_ndarray_opt v) -> Some v
      | _ -> None
    in
    let add_new acc v = if List.memq v acc then acc else v :: acc in
    lin.eqns
    |> List.concat_map (fun (e : Linear.eqn) -> List.filter_map known_tracer e.inputs)
    |> List.fold_left add_new []
    |> List.rev
  ;;

  let is_unknown = function
    | Linear.Var _ -> true
    | Known _ -> false
  ;;

  let known_values atoms =
    List.filter_map
      (function
        | Linear.Known v -> Some v
        | Var _ -> None)
      atoms
  ;;

  let call_partial_eval ~stage p xs ~known =
    match p with
    | Call { name; graph = g } ->
      let unknown = List.map not known in
      let unknown_avals = pick unknown xs |> avals in
      let known_graph, (lin, res) =
        pick known xs
        |> avals
        |> traced_aux (fun kxs ->
          let fixed = List.map (fun v -> Linear.Fixed v) kxs in
          let linear = List.map (fun a -> Linear.Linear a) unknown_avals in
          let lin, () = merge known fixed linear |> Partial_eval.trace (as_fn g) in
          let res = residuals lin in
          known_values lin.outputs @ res, (lin, res))
      in
      let ys, rs =
        call (wrap "known" name) known_graph (pick known xs)
        |> split (List.length known_graph.outputs - List.length res)
      in
      let out_unknown = List.map is_unknown lin.outputs in
      let unknown_part l =
        let rs, us = split (List.length res) l in
        let env = List.combine res rs in
        let known v = List.assq_opt v env |> Option.value ~default:v in
        pick out_unknown lin.outputs |> List.map (run_linear lin ~known us)
      in
      let dys =
        if List.exists Fun.id out_unknown
        then (
          let ug = avals rs @ unknown_avals |> traced unknown_part in
          let name = wrap "unknown" name in
          stage (Call { name; graph = ug }) (rs @ pick unknown xs))
        else []
      in
      merge out_unknown dys ys
    | _ -> stage p xs
  ;;

  let call_transpose p operands cts =
    match p with
    | Call { name; graph = g } ->
      let fixed = fixed_values operands in
      let cts = instantiate cts in
      let back =
        avals fixed @ avals cts
        |> traced (fun l ->
          let fs, cts = split (List.length fixed) l in
          transposed g operands fs cts)
      in
      let linear_cts = call (wrap "transpose" name) back (fixed @ cts) |> nonzero in
      merge (List.map is_linear operands) linear_cts (zero_like fixed)
    | _ -> Error.type_error "transpose of %s: not a call" (Prim.name p)
  ;;

  let () =
    let key = Call { name = ""; graph = empty } in
    Prim.register key ~name:"call" ~abstract_eval:call_abstract_eval ~impl:call_impl;
    Ir.register_sub_graphs key (function
      | Call { name; graph } -> [ name, graph ]
      | _ -> []);
    Jvp.register_rule key call_jvp;
    Partial_eval.register_rule key call_partial_eval;
    Transpose.register_rule key call_transpose
  ;;

  let eager : backend =
    (module struct
      let name = "eager"
      let compile f = f
    end)
  ;;

  let jit_with ~on_trace : backend =
    (module struct
      let name = "jit"

      let compile f =
        let cache = Hashtbl.create 8 in
        let trace avals =
          let g, aux = Ir.trace_aux f avals in
          on_trace g;
          let entry = Ir.closure_convert g, aux in
          (match g.captures with
           | [] -> Hashtbl.add cache avals entry
           | _ :: _ -> ());
          entry
        in
        fun xs ->
          let avals = List.map aval xs in
          let (g, captured), aux =
            match Hashtbl.find_opt cache avals with
            | Some entry -> entry
            | None -> trace avals
          in
          call "jit" g (captured @ xs), aux
      ;;
    end)
  ;;

  let jit = jit_with ~on_trace:ignore

  let compile_tree ?(backend = eager) ~input:in_tree ~output:out_tree f =
    let module B = (val backend : Backend) in
    let instances = Hashtbl.create 4 in
    let instance s =
      match Hashtbl.find_opt instances s with
      | Some c -> c
      | None ->
        let c = Tree.flat_fn in_tree out_tree f s |> B.compile in
        Hashtbl.add instances s c;
        c
    in
    fun x ->
      let xs, s = Tree.flatten in_tree x in
      let ys, out = instance s xs in
      if List.length ys <> Tree.num_leaves out
      then
        Printf.ksprintf
          invalid_arg
          "Compile: backend %s returned %d leaves, the output has %d"
          B.name
          (List.length ys)
          (Tree.num_leaves out);
      Tree.unflatten out_tree out ys
  ;;

  let compile ?backend f = compile_tree ?backend ~input:Tree.value ~output:Tree.value f
end

module Control = struct
  open Interp
  module Z = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Linear = Autodiff.Linear
  module Partial_eval = Autodiff.Partial_eval

  type Prim.t +=
    | Cond of
        { then_ : Ir.t
        ; else_ : Ir.t
        }
    | While of
        { cond : Ir.t
        ; body : Ir.t
        }

  open Util

  let is_scalar (a : Aval.t) = Array.length a.shape = 0

  let closed_aux f in_avals =
    let g, aux = Ir.trace_aux f in_avals in
    let g, captured = Ir.closure_convert g in
    g, captured, aux
  ;;

  let closed f in_avals = Ir.trace f in_avals |> Ir.closure_convert
  let split_branches (then_ : Ir.t) rest = split (List.length then_.params) rest

  let cond_abstract_eval p avals =
    match p, avals with
    | Cond { then_; else_ }, pred :: rest ->
      if not (is_scalar pred)
      then
        Error.type_error
          "cond: the predicate must be a scalar, got %s"
          (Aval.to_string pred);
      let nt = List.length then_.params
      and ne = List.length else_.params in
      if List.length rest <> nt + ne then Prim.arity_error p ~expected:(1 + nt + ne) avals;
      let at, ae = split nt rest in
      expect "cond then" ~graph:(Ir.avals then_.params) ~operands:at;
      expect "cond else" ~graph:(Ir.avals else_.params) ~operands:ae;
      let ot = Ir.avals then_.outputs
      and oe = Ir.avals else_.outputs in
      if not (List.equal Aval.equal ot oe)
      then
        Error.type_error
          "cond: the branches return (%s) and (%s)"
          (avals_str ot)
          (avals_str oe);
      ot
    | _ -> Prim.arity_error p ~expected:1 avals
  ;;

  let cond_impl p arrays =
    match p, arrays with
    | Cond { then_; else_ }, pred :: rest ->
      let at, ae = split_branches then_ rest in
      if Ndarray.to_float pred > 0. then run_arrays then_ at else run_arrays else_ ae
    | _ -> Prim.arity_error p ~expected:1 arrays
  ;;

  let jvp_graph (g : Ir.t) =
    let a = Ir.avals g.params in
    a @ a
    |> traced (fun l ->
      let xs, dxs = split (List.length a) l in
      let ys, dys, () = Jvp.jvp_flat (as_fn g) xs (nonzero dxs) in
      ys @ instantiate dys)
  ;;

  let cond_jvp p xs dxs =
    match p, xs, dxs with
    | Cond { then_; else_ }, pred :: rest, _ :: drest ->
      let at, ae = split_branches then_ rest
      and dt, de = split_branches then_ drest in
      let outs =
        bind
          (Cond { then_ = jvp_graph then_; else_ = jvp_graph else_ })
          ((pred :: (at @ instantiate dt)) @ ae @ instantiate de)
      in
      let ys, dys = split (List.length then_.outputs) outs in
      ys, nonzero dys
    | _ -> Prim.arity_error p ~expected:1 xs
  ;;

  let rec unknown_outputs (g : Ir.t) known =
    let unknown env (x : Ir.var) = Env.find_opt x.id env |> Option.value ~default:false in
    let step env (e : Ir.eqn) =
      let ins = List.map (unknown env) e.inputs in
      let outs =
        match e.prim, ins with
        | Cond { then_; else_ }, false :: rest ->
          let ut, ue = split_branches then_ rest in
          let then_unknown = unknown_outputs then_ (List.map not ut) in
          let else_unknown = unknown_outputs else_ (List.map not ue) in
          List.map2 ( || ) then_unknown else_unknown
        | Compile.Call { graph; _ }, ins -> unknown_outputs graph (List.map not ins)
        | _ -> List.map (fun _ -> List.exists Fun.id ins) e.outputs
      in
      Env.add_all (Ir.ids e.outputs) outs env
    in
    let params = Env.add_all (Ir.ids g.params) (List.map not known) Env.empty in
    let env = List.fold_left step params g.eqns in
    List.map (unknown env) g.outputs
  ;;

  let restrict (g : Ir.t) ~outputs =
    let g = { g with outputs = pick outputs g.outputs } in
    let live = { g with eqns = pick (Ir.live_mask g) g.eqns } in
    Ir.avals g.params |> traced (Ir.run live)
  ;;

  let known_part (g : Ir.t) ~known ~outputs =
    let g = restrict g ~outputs in
    pick known g.params
    |> Ir.avals
    |> traced (fun xs ->
      let unknown =
        reject known g.params |> Ir.avals |> List.map (fun a -> Linear.Linear a)
      in
      let ops = merge known (List.map (fun v -> Linear.Fixed v) xs) unknown in
      let lin, () = Partial_eval.trace (as_fn g) ops in
      lin.outputs
      |> List.map (function
        | Linear.Known v -> v
        | Linear.Var _ -> failwith "Control: a known output of a cond is unknown"))
  ;;

  let cond_partial_eval ~stage p xs ~known =
    match p, xs, known with
    | Cond { then_; else_ }, pred :: rest, true :: krest ->
      let at, ae = split_branches then_ rest
      and kt, ke = split_branches then_ krest in
      let unknown =
        List.map2 ( || ) (unknown_outputs then_ kt) (unknown_outputs else_ ke)
      in
      let outputs = List.map not unknown in
      let ys =
        if List.exists Fun.id outputs
        then
          bind
            (Cond
               { then_ = known_part then_ ~known:kt ~outputs
               ; else_ = known_part else_ ~known:ke ~outputs
               })
            ((pred :: pick kt at) @ pick ke ae)
        else []
      in
      let dys =
        if List.exists Fun.id unknown
        then
          stage
            (Cond
               { then_ = restrict then_ ~outputs:unknown
               ; else_ = restrict else_ ~outputs:unknown
               })
            xs
        else []
      in
      merge unknown dys ys
    | _ -> stage p xs
  ;;

  let cond_transpose p operands cts =
    match p, operands with
    | Cond { then_; else_ }, Linear.Fixed pred :: rest ->
      let ot, oe = split_branches then_ rest in
      let cts = instantiate cts in
      let back (g : Ir.t) ops ~before ~after =
        let fixed = fixed_values ops in
        avals fixed @ avals cts
        |> traced (fun l ->
          let fs, cts = split (List.length fixed) l in
          zeros before @ transposed g ops fs cts @ zeros after)
      in
      let outs =
        bind
          (Cond
             { then_ = back then_ ot ~before:[] ~after:(linear_avals oe)
             ; else_ = back else_ oe ~before:(linear_avals ot) ~after:[]
             })
          ((pred :: (fixed_values ot @ cts)) @ fixed_values oe @ cts)
      in
      let operand_cts =
        merge (List.map is_linear rest) (nonzero outs) (zero_like (fixed_values rest))
      in
      Z.zero_like pred :: operand_cts
    | _ -> Error.type_error "transpose of cond: the predicate is not fixed"
  ;;

  type while_operands =
    { cond_captures : value list
    ; body_captures : value list
    ; carry : value list
    }

  let while_layout cond (body : Ir.t) =
    let n = List.length body.outputs in
    List.length cond.Ir.params - n, List.length body.params - n, n
  ;;

  let split_while cond body l =
    let nc, nb, _ = while_layout cond body in
    let cond_captures, rest = split nc l in
    let body_captures, carry = split nb rest in
    cond_captures, body_captures, carry
  ;;

  let while_abstract_eval p avals =
    match p with
    | While { cond; body } ->
      let nc, nb, n = while_layout cond body in
      if nc < 0 || nb < 0 || List.length avals <> nc + nb + n
      then Prim.arity_error p ~expected:(nc + nb + n) avals;
      let ac, ab, carry = split_while cond body avals in
      expect "while cond" ~graph:(Ir.avals cond.params) ~operands:(ac @ carry);
      expect "while body" ~graph:(Ir.avals body.params) ~operands:(ab @ carry);
      (match Ir.avals cond.outputs with
       | [ a ] when is_scalar a -> ()
       | l ->
         Error.type_error "while_loop: cond must return a scalar, got (%s)" (avals_str l));
      let returned = Ir.avals body.outputs in
      if not (List.equal Aval.equal returned carry)
      then
        Error.type_error
          "while_loop: body returns (%s) for a carry of (%s)"
          (avals_str returned)
          (avals_str carry);
      carry
    | _ -> Prim.arity_error p ~expected:0 avals
  ;;

  let while_impl p arrays =
    match p with
    | While { cond; body } ->
      let ac, ab, carry = split_while cond body arrays in
      let rec loop carry =
        match run_arrays cond (ac @ carry) with
        | [ c ] when Ndarray.to_float c > 0. -> run_arrays body (ab @ carry) |> loop
        | _ -> carry
      in
      loop carry
    | _ -> Prim.arity_error p ~expected:0 arrays
  ;;

  let while_jvp p xs dxs =
    match p with
    | While { cond; body } ->
      let nc, nb, n = while_layout cond body in
      let ac, ab, carry = split_while cond body xs in
      let _, dab, dcarry = split_while cond body dxs in
      let ca = avals carry in
      let cond' =
        avals ac @ ca @ ca
        |> traced (fun l ->
          let ac, l = split nc l in
          let carry, _ = split n l in
          Ir.run cond (ac @ carry))
      in
      let body' =
        avals ab @ avals ab @ ca @ ca
        |> traced (fun l ->
          let ab, l = split nb l in
          let dab, l = split nb l in
          let carry, dcarry = split n l in
          let dxs = nonzero (dab @ dcarry) in
          let ys, dys, () = Jvp.jvp_flat (as_fn body) (ab @ carry) dxs in
          ys @ instantiate dys)
      in
      let outs =
        bind
          (While { cond = cond'; body = body' })
          (ac @ (ab @ instantiate dab) @ carry @ instantiate dcarry)
      in
      let ys, dys = split n outs in
      ys, nonzero dys
    | _ -> Prim.arity_error p ~expected:0 xs
  ;;

  let while_partial_eval ~stage:_ _ _ ~known:_ =
    Error.missing_rule
      "vjp of while_loop: reverse mode needs a fixed trip count; use forward mode \
       (Ad.jvp, Ad.jacfwd) or an OCaml loop"
  ;;

  let () =
    let cond_key = Cond { then_ = empty; else_ = empty }
    and while_key = While { cond = empty; body = empty } in
    Prim.register cond_key ~name:"cond" ~abstract_eval:cond_abstract_eval ~impl:cond_impl;
    Prim.register
      while_key
      ~name:"while"
      ~abstract_eval:while_abstract_eval
      ~impl:while_impl;
    Ir.register_sub_graphs cond_key (function
      | Cond { then_; else_ } -> [ "then", then_; "else", else_ ]
      | _ -> []);
    Ir.register_sub_graphs while_key (function
      | While { cond; body } -> [ "cond", cond; "body", body ]
      | _ -> []);
    Jvp.register_rule cond_key cond_jvp;
    Partial_eval.register_rule cond_key cond_partial_eval;
    Autodiff.Transpose.register_rule cond_key cond_transpose;
    Jvp.register_rule while_key while_jvp;
    Partial_eval.register_rule while_key while_partial_eval
  ;;

  let cond_tree ~input:in_tree ~output:out_tree pred ~then_ ~else_ x =
    let xs, s = Tree.flatten in_tree x in
    let branch f = closed_aux (Tree.flat_fn in_tree out_tree f s) (avals xs) in
    let gt, ct, out_t = branch then_
    and ge, ce, out_e = branch else_ in
    if not (Tree.equal_structure out_t out_e)
    then Error.type_error "cond: the branches return different structures";
    bind (Cond { then_ = gt; else_ = ge }) ((pred :: (ct @ xs)) @ ce @ xs)
    |> Tree.unflatten out_tree out_t
  ;;

  let cond pred ~then_ ~else_ x =
    cond_tree ~input:Tree.value ~output:Tree.value pred ~then_ ~else_ x
  ;;

  let while_loop_tree ~carry:tree ~cond ~body init =
    let xs, s = Tree.flatten tree init in
    let of_leaves leaves = Tree.unflatten tree s leaves in
    let to_leaves carry =
      let ys, s' = Tree.flatten tree carry in
      if not (Tree.equal_structure s' s)
      then Error.type_error "while_loop: body changed the structure of the carry";
      ys
    in
    let gc, cc = avals xs |> closed (fun leaves -> [ of_leaves leaves |> cond ]) in
    let gb, bc =
      avals xs |> closed (fun leaves -> of_leaves leaves |> body |> to_leaves)
    in
    bind (While { cond = gc; body = gb }) (cc @ bc @ xs) |> of_leaves
  ;;

  let while_loop ~cond ~body x = while_loop_tree ~carry:Tree.value ~cond ~body x
end
