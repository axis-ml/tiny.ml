open Common
open Lang

let same_type n p avals =
  match avals with
  | x :: rest when List.length avals = n ->
    if List.exists (fun y -> not (Aval.equal x y)) rest
    then
      Error.type_error
        "%s: operands must have the same type, got %s"
        (Prim.name p)
        (List.map Aval.to_string avals |> String.concat " and ");
    [ x ]
  | _ -> Prim.arity_error p ~expected:n avals
;;

let register p ~name ~n ~impl ~partials =
  Prim.register p ~name ~abstract_eval:(same_type n) ~impl:(fun _ arrays ->
    [ impl arrays ]);
  Autodiff.Derivatives.register_elementwise p ~partials:(fun _ xs _ -> partials xs);
  Batching.Batching.register_rule p Batching.Batching.elementwise_rule
;;

let unary ~name ~impl ~deriv =
  let module M = struct
    type Prim.t += P
  end
  in
  register
    M.P
    ~name
    ~n:1
    ~impl:(function
      | [ x ] -> Ndarray.map impl x
      | l -> Prim.arity_error M.P ~expected:1 l)
    ~partials:(List.map (fun x -> Some (deriv x)));
  fun x -> Interp.bind1 M.P [ x ]
;;

let binary ~name ~impl ~deriv =
  let module M = struct
    type Prim.t += P
  end
  in
  register
    M.P
    ~name
    ~n:2
    ~impl:(function
      | [ x; y ] -> Ndarray.map2 impl x y
      | l -> Prim.arity_error M.P ~expected:2 l)
    ~partials:(function
      | [ x; y ] ->
        let dx, dy = deriv x y in
        [ Some dx; Some dy ]
      | l -> Prim.arity_error M.P ~expected:2 l);
  fun x y -> Interp.bind1 M.P (Ops.broadcast_operands M.P [ x; y ])
;;
