open Common

(** Primitives: an open type, extensible from outside the library. See
    [examples/extending.ml]. *)
module Prim : sig
  type t = ..

  (** Registers (or replaces) [p]'s constructor; its arguments are ignored.
      Raises [Invalid_argument] if [name] is taken. [abstract_eval] raises
      [Type_error] on bad operands. *)
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

  (** A transformation's rules, keyed by primitive constructor. *)
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

(** Values, interpreters and dispatch. An interpreter is an effect handler
    installed by {!handle}; its own binds go to the interpreter below, with
    eval at the bottom. *)
module Interp : sig
  type value

  (** An interpreter whose tracers carry a ['p]. *)
  type 'p t

  val of_ndarray : Ndarray.t -> value
  val aval : value -> Aval.t
  val describe : value -> string

  (** Raises [Concretization_error] if some interpreter has no data (e.g.
      staging), [Escaped_tracer] if the tracer outlived its {!handle}. *)
  val to_ndarray : value -> Ndarray.t

  val to_ndarray_opt : value -> Ndarray.t option

  (** [false] for a tracer whose {!handle} returned. *)
  val alive : value -> bool

  (** [concrete payload] is the value under a tracer, if any. *)
  val create
    :  name:string
    -> ?concrete:('p -> value option)
    -> ('p t -> Prim.t -> value list -> value list)
    -> 'p t

  val name : _ t -> string
  val id : _ t -> int
  val tracer : 'p t -> Aval.t -> 'p -> value

  (** [None] for arrays and tracers of other interpreters. *)
  val payload : 'p t -> value -> 'p option

  val bind : Prim.t -> value list -> value list
  val bind1 : Prim.t -> value list -> value

  (** Not domain-safe: tracer liveness is a counter on [i]. *)
  val handle : _ t -> (unit -> 'a) -> 'a

  (** e.g. ["eval < jvp#3 < jvp#7"]. *)
  val stack_to_string : unit -> string
end

(** The built-in primitives; everything else in [Ops] is composed from
    them. *)
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

(** Array ops; they work under every interpreter. Binary ops broadcast like
    NumPy; comparisons return 1.0/0.0 masks. *)
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

  (** Never raises: describes tracers. *)
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

  (** [select p a b] is [a] where [p > 0], else [b]. *)
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

  (** Inserts dims at [axes], repeating the data. *)
  val broadcast : shape:int array -> axes:int list -> value -> value

  val broadcast_to : shape:int array -> value -> value
  val concatenate : ?axis:int -> value list -> value

  (** [[start, stop)] along [axis]; negative indices count from the end. *)
  val slice : ?axis:int -> start:int -> stop:int -> value -> value

  val stack : ?axis:int -> value list -> value

  (** Rank 1 and 2 operands. *)
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

(** Structured arguments for transformations (JAX's pytrees). *)
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

  (** A tree without its leaves (JAX's treedef); comparable with [=]. *)
  type structure

  val flatten : 'a t -> 'a -> value list * structure

  (** Raises [Type_error] if the leaves do not fit. *)
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

  (** Raises [Type_error] on different structures. *)
  val map2 : 'a t -> (value -> value -> value) -> 'a -> 'a -> 'a
end
