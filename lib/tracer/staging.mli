open Common
open Lang

(** A traced DAG (JAX's jaxpr). Only types are known while tracing. *)
module Ir : sig
  type value := Interp.value

  type var = private
    { id : int
    ; aval : Aval.t
    }

  type eqn = private
    { prim : Prim.t
    ; inputs : var list
    ; outputs : var list
    }

  (** [eqns] in eval order; ids [< num_vars]. Free arrays are [consts], free
      outer tracers are [captures] (valid only while their transformation
      runs; see {!closure_convert}). *)
  type t = private
    { params : var list
    ; captures : (var * value) list
    ; consts : (var * Ndarray.t) list
    ; eqns : eqn list
    ; outputs : var list
    ; num_vars : int
    }

  val trace : (value list -> value list) -> Aval.t list -> t
  val trace_aux : (value list -> value list * 'a) -> Aval.t list -> t * 'a

  (** Traces at [x]'s type. *)
  val make : (value -> value) -> value -> t

  val make_tree : input:'a Tree.t -> output:'b Tree.t -> ('a -> 'b) -> 'a -> t

  (** Captures become leading params; returns their values. *)
  val closure_convert : t -> t * value list

  (** Raises [Failure] if ill-formed. *)
  val invariant : t -> unit

  (** Works under every interpreter. *)
  val run : t -> value list -> value list

  val run1 : t -> value -> value

  (** {2 Inspection} *)

  (** By var id. *)
  val producers : t -> eqn option array

  (** By var id. *)
  val users : t -> eqn list array

  (** Eqns no output needs (still run). *)
  val dead : t -> eqn list

  (** e.g. a cond's branches. *)
  val sub_graphs : Prim.t -> (string * t) list

  (** Lets [invariant], [to_string], [to_dot] see inside. *)
  val register_sub_graphs : Prim.t -> (Prim.t -> (string * t) list) -> unit

  val to_string : t -> string
  val pp : Format.formatter -> t -> unit

  (** Graphviz; dead eqns dashed. *)
  val to_dot : t -> string

  (** {2 Construction}

      Unchecked; use {!invariant}. *)
  module Unchecked : sig
    val var : id:int -> Aval.t -> var
    val eqn : Prim.t -> inputs:var list -> outputs:var list -> eqn

    val graph
      :  ?captures:(var * value) list
      -> ?consts:(var * Ndarray.t) list
      -> params:var list
      -> eqns:eqn list
      -> outputs:var list
      -> num_vars:int
      -> unit
      -> t
  end
end

(** For rules of graph-carrying primitives. Internal. *)
module Util : sig
  type value := Interp.value

  (** Raises [Failure] on captures. *)
  val traced : (value list -> value list) -> Aval.t list -> Ir.t

  val traced_aux : (value list -> value list * 'a) -> Aval.t list -> Ir.t * 'a

  (** Placeholder for registration keys. *)
  val empty : Ir.t

  (** For [Jvp.jvp_flat] and the like. *)
  val as_fn : Ir.t -> value list -> value list * unit
end

(** Runs a function through a backend. *)
module Compile : sig
  type value := Interp.value

  module type Backend = sig
    val name : string

    (** Once per input structure. *)
    val compile : (value list -> value list * 'aux) -> value list -> value list * 'aux
  end

  type backend = (module Backend)

  (** A closed graph called as one op (JAX's pjit). *)
  type Prim.t +=
    | Call of
        { name : string
        ; graph : Ir.t
        }

  val eager : backend

  (** Caches a [Call] per input avals. Free arrays are baked in; graphs
      capturing tracers are retraced each call. *)
  val jit : backend

  val jit_with : on_trace:(Ir.t -> unit) -> backend

  (** [backend] defaults to {!eager}. *)
  val compile : ?backend:backend -> (value -> value) -> value -> value

  val compile_tree
    :  ?backend:backend
    -> input:'a Tree.t
    -> output:'b Tree.t
    -> ('a -> 'b)
    -> 'a
    -> 'b
end

(** Control flow on data; works under staging, unlike OCaml [if]. *)
module Control : sig
  type value := Interp.value

  (** Cond operands: predicate, [then_] inputs, [else_] inputs. *)
  type Prim.t +=
    | Cond of
        { then_ : Ir.t
        ; else_ : Ir.t
        }
    | While of
        { cond : Ir.t
        ; body : Ir.t
        }
          (** Operands: [cond] captures, [body] captures, carry. *)

  (** [then_ x] if scalar [pred > 0], else [else_ x]; same result types. *)
  val cond : value -> then_:(value -> value) -> else_:(value -> value) -> value -> value

  val cond_tree
    :  input:'a Tree.t
    -> output:'b Tree.t
    -> value
    -> then_:('a -> 'b)
    -> else_:('a -> 'b)
    -> 'a
    -> 'b

  (** While [cond carry > 0]. No [vjp] ([Missing_rule]). *)
  val while_loop : cond:(value -> value) -> body:(value -> value) -> value -> value

  val while_loop_tree
    :  carry:'a Tree.t
    -> cond:('a -> value)
    -> body:('a -> 'a)
    -> 'a
    -> 'a
end
