open Lang

(** Batching (JAX's vmap). A batched value has one extra dim, its batch dim. *)
module Batching : sig
  type value := Interp.value

  (** Operands and their batch dims ([None]: unbatched) to outputs and theirs. *)
  type rule =
    Prim.t -> size:int -> value list -> int option list -> value list * int option list

  val register_rule : Prim.t -> rule -> unit

  (** Registered primitives with no batching rule. *)
  val missing_rules : unit -> string list

  (** Moves dim [from] to position [to_], keeping the others in order. *)
  val move_axis : from:int -> to_:int -> value -> value

  (** Batch dim to 0; broadcasts unbatched values. *)
  val to_front : size:int -> value -> int option -> value

  (** Rule for same-shape elementwise ops. *)
  val elementwise_rule : rule

  (** [vmap_flat ~size f args batch_dims] is [(outputs, their batch dims, aux)]. *)
  val vmap_flat
    :  size:int
    -> (value list -> value list * 'aux)
    -> value list
    -> int option list
    -> value list * int option list * 'aux
end

(** Write [f] for one example, run it on a batch. Nests with other
    transformations and itself.

    {[
      let xs = Ops.array ~shape:[| 4; 3 |] data in
      Vmap.vmap (fun x -> Ops.reduce_sum x) xs  (* shape [4] *)
    ]} *)
module Vmap : sig
  type value := Interp.value

  (** Maps [f] over [in_axis] of [x], stacks along [out_axis] (both default
      0, negative from the end). *)
  val vmap : ?in_axis:int -> ?out_axis:int -> (value -> value) -> value -> value

  (** [in_axes]: one entry per leaf of [input], [None] means shared (not
      batched, e.g. the weights). Without it every leaf uses [in_axis].
      Raises [Type_error] if the lengths differ. *)
  val vmap_tree
    :  ?in_axes:int option list
    -> ?in_axis:int
    -> ?out_axis:int
    -> input:'a Tree.t
    -> output:'b Tree.t
    -> ('a -> 'b)
    -> 'a
    -> 'b
end
