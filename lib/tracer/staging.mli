open Common
open Lang

(** A function traced once into a DAG of primitive applications (JAX's
    jaxpr). Tracers carry only types, so [f] cannot read data. *)
module Ir : sig
  type value := Interp.value

  type var =
    { id : int
    ; aval : Aval.t
    }

  type eqn =
    { prim : Prim.t
    ; inputs : var list
    ; outputs : var list
    }

  (** [eqns] are in evaluation order. Var ids are [0 .. num_vars - 1].

      What [f] reads without taking it as an argument is a [const] if it is
      an array, and a [capture] if it is a tracer of an enclosing
      transformation. A capture is only valid while that transformation
      runs: {!run} raises [Escaped_tracer] after, and {!closure_convert}
      turns captures into params. *)
  type t =
    { params : var list
    ; captures : (var * value) list
    ; consts : (var * Ndarray.t) list
    ; eqns : eqn list
    ; outputs : var list
    ; num_vars : int
    }

  val trace : (value list -> value list) -> Aval.t list -> t
  val trace_aux : (value list -> value list * 'a) -> Aval.t list -> t * 'a

  (** Traces at the types of [x]'s leaves; the data is unused. *)
  val make : 'a Tree.t -> 'b Tree.t -> ('a -> 'b) -> 'a -> t

  val make1 : (value -> value) -> value -> t

  (** [g'] takes [captures @ params]; returns the captured values. *)
  val closure_convert : t -> t * value list

  (** Raises [Failure] unless [g] is well formed. *)
  val invariant : t -> unit

  (** Binds each eqn, so it works under every interpreter. *)
  val run : t -> value list -> value list

  val run1 : t -> value -> value

  (** {2 Inspection} *)

  (** Indexed by var id: the eqn that defines it, if any. *)
  val producers : t -> eqn option array

  (** Indexed by var id: the eqns that read it, in order. *)
  val users : t -> eqn list array

  (** Eqns no output depends on. They still run: there are no passes yet. *)
  val dead : t -> eqn list

  (** The graphs a primitive carries (e.g. the branches of a cond). *)
  val sub_graphs : Prim.t -> (string * t) list

  (** For primitives with graph params, so that [invariant], [to_string]
      and [to_dot] look inside them. *)
  val register_sub_graphs : Prim.t -> (Prim.t -> (string * t) list) -> unit

  (** Sub-graphs are printed indented under their eqn. *)
  val to_string : t -> string

  val pp : Format.formatter -> t -> unit

  (** Graphviz source. Dead eqns are dashed, sub-graphs are clusters. *)
  val to_dot : t -> string
end

(** Runs a function through a backend on flat leaves. *)
module Compile : sig
  type value := Interp.value

  module type Backend = sig
    val name : string

    (** Called once per input structure; a caching backend keys on input
        avals and returns the extra result [f] gave for that key. *)
    val compile : (value list -> value list * 'aux) -> value list -> value list * 'aux
  end

  type backend = (module Backend)

  val eager : backend

  (** Traces once per list of input avals and runs the cached graph.
      Arrays [f] reads from outside are baked in at trace time; a graph
      that captures a tracer is not cached, so [f] is traced on each call. *)
  val jit : backend

  val jit_with : on_trace:(Ir.t -> unit) -> backend

  (** [backend] defaults to {!eager}. *)
  val compile : ?backend:backend -> 'a Tree.t -> 'b Tree.t -> ('a -> 'b) -> 'a -> 'b

  val compile1 : ?backend:backend -> (value -> value) -> value -> value
end

(** Control flow on data. Both branches (or the loop's cond and body) are
    traced into sub-graphs, so these work under every interpreter,
    including staging, where an OCaml [if] on a value cannot. *)
module Control : sig
  type value := Interp.value

  (** Operands: the predicate, then the inputs of [then_], then those of
      [else_]. Each branch takes what it captured, then the arguments. *)
  type Prim.t +=
    | Cond of
        { then_ : Ir.t
        ; else_ : Ir.t
        }
    | While of
        { cond : Ir.t
        ; body : Ir.t
        }
          (** Operands: [cond]'s captures, [body]'s captures, the carry. *)

  (** [then_ x] if the scalar [pred] is [> 0], else [else_ x]. Both branches
      are traced and must return the same types. *)
  val cond
    :  'a Tree.t
    -> 'b Tree.t
    -> value
    -> then_:('a -> 'b)
    -> else_:('a -> 'b)
    -> 'a
    -> 'b

  val cond1 : value -> then_:(value -> value) -> else_:(value -> value) -> value -> value

  (** Runs [body] while [cond carry > 0]. [body] keeps the carry's types.
      Forward-differentiable only: [vjp] raises [Missing_rule]. *)
  val while_loop : 'a Tree.t -> cond:('a -> value) -> body:('a -> 'a) -> 'a -> 'a

  val while_loop1 : cond:(value -> value) -> body:(value -> value) -> value -> value
end
