open Common

(** Open type of primitives; see [examples/extending.ml]. *)
module Prim : sig
  type t = ..

  (** Keyed by constructor, arguments ignored. Raises [Invalid_argument] if
      [name] is taken. *)
  val register
    :  t
    -> name:string
    -> abstract_eval:(t -> Aval.t list -> Aval.t list)
    -> impl:(t -> Ndarray.t list -> Ndarray.t list)
    -> unit

  val name : t -> string
  val abstract_eval : t -> Aval.t list -> Aval.t list
  val impl : t -> Ndarray.t list -> Ndarray.t list
  val names : unit -> string list
  val arity_error : t -> expected:int -> 'a list -> 'b

  (** Rules keyed by constructor. *)
  module Table : sig
    type prim := t
    type 'a t

    val create : string -> 'a t
    val add : 'a t -> prim -> 'a -> unit

    (** Raises [Missing_rule]. *)
    val find : 'a t -> prim -> 'a

    val find_opt : 'a t -> prim -> 'a option
    val missing : 'a t -> string list
  end
end

(** Interpreters are effect handlers ({!handle}) stacked over eval. *)
module Interp : sig
  type value

  (** Tracers carry a ['p]. *)
  type 'p t

  val of_ndarray : Ndarray.t -> value
  val aval : value -> Aval.t
  val describe : value -> string

  (** Raises [Concretization_error] or [Escaped_tracer]. *)
  val to_ndarray : value -> Ndarray.t

  val to_ndarray_opt : value -> Ndarray.t option

  (** [false] once the tracer's {!handle} returned. *)
  val alive : value -> bool

  (** [concrete]: the value under a tracer, if any. *)
  val create
    :  name:string
    -> ?concrete:('p -> value option)
    -> ('p t -> Prim.t -> value list -> value list)
    -> 'p t

  val name : _ t -> string
  val id : _ t -> int
  val tracer : 'p t -> Aval.t -> 'p -> value

  (** [None] if not this interpreter's tracer. *)
  val payload : 'p t -> value -> 'p option

  val bind : Prim.t -> value list -> value list
  val bind1 : Prim.t -> value list -> value

  (** Not domain-safe. *)
  val handle : _ t -> (unit -> 'a) -> 'a

  (** e.g. ["eval < jvp#3 < jvp#7"]. *)
  val stack_to_string : unit -> string
end

(** Built-in primitives; [Ops] is built on them. *)
module Builtin : sig
  type Prim.t +=
    | Add
    | Sub
    | Mul
    | Div
    | Max
    | Greater
    | Less
    | Neg
    | Sin
    | Cos
    | Exp
    | Log
    | Sqrt
    | Integer_pow of int
    | Select
    | Dot (** [[m; k]] by [[k; n]] *)
    | Reduce_sum of int list
    | Reduce_max of int list
    | Broadcast of
        { shape : int array
        ; axes : int list
        }
          (** inserts dims at [axes]; never stretches *)
    | Transpose of int array
    | Reshape of int array
    | Concatenate of int
    | Slice of
        { axis : int
        ; start : int
        ; stop : int
        }

  val all : Prim.t list
end

(** Binary ops broadcast like NumPy; comparisons return 1.0/0.0. *)
module Ops : sig
  type value = Interp.value

  val scalar : float -> value
  val array : ?shape:int array -> float array -> value
  val full : int array -> float -> value
  val zeros : int array -> value
  val ones : int array -> value
  val zeros_like : value -> value
  val ones_like : value -> value
  val arange : int -> value
  val linspace : float -> float -> int -> value
  val eye : int -> value
  val of_ndarray : Ndarray.t -> value
  val shape : value -> int array
  val dtype : value -> Aval.dtype
  val rank : value -> int
  val size : value -> int
  val to_float : value -> float
  val to_ndarray : value -> Ndarray.t
  val to_array : value -> float array
  val allclose : ?rtol:float -> ?atol:float -> value -> value -> bool

  (** Never raises. *)
  val to_string : value -> string

  val describe : value -> string
  val pp : Format.formatter -> value -> unit
  val add : value -> value -> value
  val sub : value -> value -> value
  val mul : value -> value -> value
  val div : value -> value -> value
  val maximum : value -> value -> value
  val minimum : value -> value -> value
  val neg : value -> value
  val sin : value -> value
  val cos : value -> value
  val exp : value -> value
  val log : value -> value
  val sqrt : value -> value
  val integer_pow : exponent:int -> value -> value
  val pow : exponent:float -> value -> value
  val square : value -> value
  val abs : value -> value
  val relu : value -> value
  val logistic : value -> value
  val tanh : value -> value
  val tan : value -> value
  val sinh : value -> value
  val cosh : value -> value
  val asinh : value -> value
  val acosh : value -> value
  val atanh : value -> value
  val greater : value -> value -> value
  val less : value -> value -> value
  val greater_equal : value -> value -> value
  val less_equal : value -> value -> value
  val equal : value -> value -> value
  val not_equal : value -> value -> value

  (** [a] where [p > 0], else [b]. *)
  val select : value -> value -> value -> value

  (** [axes] default to all. *)
  val reduce_sum : ?axes:int list -> ?keepdims:bool -> value -> value

  val reduce_max : ?axes:int list -> ?keepdims:bool -> value -> value
  val reduce_min : ?axes:int list -> ?keepdims:bool -> value -> value
  val mean : ?axes:int list -> ?keepdims:bool -> value -> value
  val reshape : shape:int array -> value -> value
  val expand_dims : axes:int list -> value -> value
  val squeeze : ?axes:int list -> value -> value
  val transpose : ?perm:int array -> value -> value

  (** Inserts dims at [axes]. *)
  val broadcast : shape:int array -> axes:int list -> value -> value

  val broadcast_to : shape:int array -> value -> value

  (** To a common shape; [p] names errors. *)
  val broadcast_operands : Prim.t -> value list -> value list

  val concatenate : ?axis:int -> value list -> value

  (** From [start] to [stop] (exclusive); negative from the end. *)
  val slice : ?axis:int -> start:int -> stop:int -> value -> value

  val stack : ?axis:int -> value list -> value

  (** Rank 1 or 2. *)
  val dot : value -> value -> value

  val matmul : value -> value -> value

  module Infix : sig
    val ( + ) : value -> value -> value
    val ( - ) : value -> value -> value
    val ( * ) : value -> value -> value
    val ( / ) : value -> value -> value
    val ( ~- ) : value -> value
    val ( ** ) : value -> float -> value
    val ( > ) : value -> value -> value
    val ( < ) : value -> value -> value
    val ( >= ) : value -> value -> value
    val ( <= ) : value -> value -> value
  end

  include module type of Infix
end

(** JAX's pytrees. *)
module Tree : sig
  type value := Interp.value
  type 'a t

  val value : value t
  val unit : unit t
  val pair : 'a t -> 'b t -> ('a * 'b) t
  val triple : 'a t -> 'b t -> 'c t -> ('a * 'b * 'c) t
  val list : 'a t -> 'a list t
  val array : 'a t -> 'a array t
  val conv : to_tree:('b -> 'a) -> of_tree:('a -> 'b) -> 'a t -> 'b t

  (** Treedef. *)
  type structure

  val equal_structure : structure -> structure -> bool
  val flatten : 'a t -> 'a -> value list * structure

  (** Raises [Type_error]. *)
  val unflatten : 'a t -> structure -> value list -> 'a

  val leaves : 'a t -> 'a -> value list
  val structure : 'a t -> 'a -> structure

  val flat_fn
    :  'a t
    -> 'b t
    -> ('a -> 'b)
    -> structure
    -> value list
    -> value list * structure

  val num_leaves : structure -> int
  val map : 'a t -> (value -> value) -> 'a -> 'a

  (** Raises [Type_error]. *)
  val map2 : 'a t -> (value -> value -> value) -> 'a -> 'a -> 'a
end
