open Common

(** Open type of primitives; see [examples/extending.ml]. A primitive is one
    op the tracer knows (add, sin, dot, ...); every transformation is a set of
    rules keyed by primitive. *)
module Prim : sig
  type t = ..

  (** Keyed by constructor, arguments ignored. [abstract_eval] computes output
      types from input types; [impl] computes on real arrays. Raises
      [Invalid_argument] if [name] is taken. *)
  val register
    :  t
    -> name:string
    -> abstract_eval:(t -> Aval.t list -> Aval.t list)
    -> impl:(t -> Ndarray.t list -> Ndarray.t list)
    -> unit

  (** The registered name, e.g. ["add"]. *)
  val name : t -> string

  (** Output types from input types. *)
  val abstract_eval : t -> Aval.t list -> Aval.t list

  (** Runs on concrete arrays. *)
  val impl : t -> Ndarray.t list -> Ndarray.t list

  (** All registered names, sorted. *)
  val names : unit -> string list

  (** Raises [Type_error] "expected [expected] operand(s)". *)
  val arity_error : t -> expected:int -> 'a list -> 'b

  (** Rules keyed by constructor. *)
  module Table : sig
    type prim := t
    type 'a t

    (** [create name]; [name] (e.g. ["jvp"]) appears in [Missing_rule]. *)
    val create : string -> 'a t

    (** Replaces any rule already there. *)
    val add : 'a t -> prim -> 'a -> unit

    (** Raises [Missing_rule]. *)
    val find : 'a t -> prim -> 'a

    val find_opt : 'a t -> prim -> 'a option

    (** Registered primitives with no rule here, sorted. *)
    val missing : 'a t -> string list
  end
end

(** Interpreters are effect handlers ({!handle}) stacked over eval. Each op
    goes to the innermost active interpreter; with none, it just computes. *)
module Interp : sig
  (** An array, or a tracer standing in for one. *)
  type value

  (** Tracers carry a ['p]. *)
  type 'p t

  (** Wraps an array. *)
  val of_ndarray : Ndarray.t -> value

  (** Its shape and dtype; always known. *)
  val aval : value -> Aval.t

  (** e.g. ["array f64[3]"] or ["jvp tracer #3 f64[3]"]. *)
  val describe : value -> string

  (** Raises [Concretization_error] or [Escaped_tracer]. *)
  val to_ndarray : value -> Ndarray.t

  (** [None] for any tracer. *)
  val to_ndarray_opt : value -> Ndarray.t option

  (** [false] once the tracer's {!handle} returned. *)
  val alive : value -> bool

  (** [create ~name process]: [process self prim args] handles one op whose
      args include this interpreter's tracers. [concrete]: the value under a
      tracer, if any. *)
  val create
    :  name:string
    -> ?concrete:('p -> value option)
    -> ('p t -> Prim.t -> value list -> value list)
    -> 'p t

  val name : _ t -> string

  (** Unique per [create]; shown as [name#id]. *)
  val id : _ t -> int

  (** A new tracer of this interpreter with the given type and payload. *)
  val tracer : 'p t -> Aval.t -> 'p -> value

  (** [None] if not this interpreter's tracer. *)
  val payload : 'p t -> value -> 'p option

  (** Applies a primitive: the entry point for every op. *)
  val bind : Prim.t -> value list -> value list

  (** [bind] for single-output primitives. *)
  val bind1 : Prim.t -> value list -> value

  (** Runs [f] with this interpreter on top of the stack. Not domain-safe. *)
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

  (** Every constructor above. *)
  val all : Prim.t list
end

(** NumPy-like ops; they work on arrays and tracers alike. Binary ops
    broadcast like NumPy; comparisons return 1.0/0.0. *)
module Ops : sig
  type value = Interp.value

  (** {2 Creating} *)

  (** Rank 0. *)
  val scalar : float -> value

  (** Flat data; [shape] defaults to [[|length|]]. Raises [Type_error] if the
      sizes differ. *)
  val array : ?shape:int array -> float array -> value

  val full : int array -> float -> value
  val zeros : int array -> value
  val ones : int array -> value

  (** Same shape as the argument. *)
  val zeros_like : value -> value

  val ones_like : value -> value

  (** [[0.; 1.; ...; n-1.]]. *)
  val arange : int -> value

  (** [n] points from [lo] to [hi], both included. *)
  val linspace : float -> float -> int -> value

  (** [n]x[n] identity. *)
  val eye : int -> value

  val of_ndarray : Ndarray.t -> value

  (** {2 Inspecting}

      Shape queries work on tracers; reading data does not. *)

  val shape : value -> int array
  val dtype : value -> Aval.dtype
  val rank : value -> int

  (** Number of elements. *)
  val size : value -> int

  (** The value of a scalar. Raises [Type_error] if not rank 0, or as
      {!to_ndarray}. *)
  val to_float : value -> float

  (** Raises [Concretization_error] on a tracer (e.g. inside [jit]). *)
  val to_ndarray : value -> Ndarray.t

  (** Flat data in C order. *)
  val to_array : value -> float array

  (** See {!Ndarray.allclose}. *)
  val allclose : ?rtol:float -> ?atol:float -> value -> value -> bool

  (** The data, or {!describe} for a tracer. Never raises. *)
  val to_string : value -> string

  (** Type only, e.g. ["array f64[3]"]. *)
  val describe : value -> string

  val pp : Format.formatter -> value -> unit

  (** {2 Elementwise} *)

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

  (** Natural log. *)
  val log : value -> value

  val sqrt : value -> value

  (** [x ** n] for an int [n], by repeated multiplication. *)
  val integer_pow : exponent:int -> value -> value

  (** [x ** p]; uses {!integer_pow} when [p] is a small integer, else
      [exp (p * log x)]. *)
  val pow : exponent:float -> value -> value

  val square : value -> value
  val abs : value -> value

  (** [max x 0]. *)
  val relu : value -> value

  (** Sigmoid: [1 / (1 + exp (-x))]. *)
  val logistic : value -> value

  val tanh : value -> value
  val tan : value -> value
  val sinh : value -> value
  val cosh : value -> value
  val asinh : value -> value
  val acosh : value -> value
  val atanh : value -> value

  (** {2 Comparisons}

      1.0 where true, 0.0 elsewhere. Their derivative is zero. *)

  val greater : value -> value -> value
  val less : value -> value -> value
  val greater_equal : value -> value -> value
  val less_equal : value -> value -> value
  val equal : value -> value -> value
  val not_equal : value -> value -> value

  (** [select p a b]: [a] where [p > 0], else [b]. Elementwise [if]. *)
  val select : value -> value -> value -> value

  (** {2 Reductions}

      [axes] default to all; [keepdims] (default [false]) keeps them as size 1. *)

  (** e.g. [reduce_sum ~axes:[0]] sums each column of a matrix. *)
  val reduce_sum : ?axes:int list -> ?keepdims:bool -> value -> value

  val reduce_max : ?axes:int list -> ?keepdims:bool -> value -> value
  val reduce_min : ?axes:int list -> ?keepdims:bool -> value -> value
  val mean : ?axes:int list -> ?keepdims:bool -> value -> value

  (** {2 Shapes}

      Negative axes count from the end. *)

  (** Same data, new shape (same number of elements). *)
  val reshape : shape:int array -> value -> value

  (** Inserts size-1 dims at [axes]. *)
  val expand_dims : axes:int list -> value -> value

  (** Removes size-1 dims at [axes] (default: all of them). *)
  val squeeze : ?axes:int list -> value -> value

  (** [perm] defaults to reversing the dims (matrix transpose). *)
  val transpose : ?perm:int array -> value -> value

  (** Inserts dims at [axes]. *)
  val broadcast : shape:int array -> axes:int list -> value -> value

  (** NumPy's [broadcast_to]: pads leading dims and repeats size-1 dims. *)
  val broadcast_to : shape:int array -> value -> value

  (** To a common shape; [p] names errors. *)
  val broadcast_operands : Prim.t -> value list -> value list

  (** Joins along an existing [axis] (default 0). *)
  val concatenate : ?axis:int -> value list -> value

  (** From [start] to [stop] (exclusive); negative from the end. *)
  val slice : ?axis:int -> start:int -> stop:int -> value -> value

  (** Joins along a new [axis] (default 0): [n] arrays of shape [s] give
      [n :: s]. *)
  val stack : ?axis:int -> value list -> value

  (** {2 Linear algebra} *)

  (** Rank 1 or 2: vector dot, matrix-vector or matrix-matrix. *)
  val dot : value -> value -> value

  (** Same as {!dot}. *)
  val matmul : value -> value -> value

  (** Operators; [open Ops] or [Ops.(x * y + z)]. [**] is {!pow}. *)
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

(** JAX's pytrees: a ['a t] describes how to find the [value]s (leaves)
    inside an OCaml value of type ['a], so transformations can take records,
    tuples and lists, not just one array.

    {[
      let params = Tree.(pair value (list value))
    ]} *)
module Tree : sig
  type value := Interp.value
  type 'a t

  (** One leaf. *)
  val value : value t

  (** No leaves. *)
  val unit : unit t

  val pair : 'a t -> 'b t -> ('a * 'b) t
  val triple : 'a t -> 'b t -> 'c t -> ('a * 'b * 'c) t

  (** Any length; the length is part of the {!type-structure}. *)
  val list : 'a t -> 'a list t

  val array : 'a t -> 'a array t

  (** For your own types (e.g. records): convert to and from a type that
      already has a description. [[@@deriving tree]] writes this for you. *)
  val conv : to_tree:('b -> 'a) -> of_tree:('a -> 'b) -> 'a t -> 'b t

  (** Treedef: the shape of a value with its leaves removed. *)
  type structure

  val equal_structure : structure -> structure -> bool

  (** Leaves in order, and the structure to rebuild with. *)
  val flatten : 'a t -> 'a -> value list * structure

  (** Inverse of {!flatten}. Raises [Type_error]. *)
  val unflatten : 'a t -> structure -> value list -> 'a

  (** [fst (flatten t x)]. *)
  val leaves : 'a t -> 'a -> value list

  (** [snd (flatten t x)]. *)
  val structure : 'a t -> 'a -> structure

  (** [f] as a function on leaf lists: unflattens with the given input
      structure, calls [f], flattens the result. *)
  val flat_fn
    :  'a t
    -> 'b t
    -> ('a -> 'b)
    -> structure
    -> value list
    -> value list * structure

  val num_leaves : structure -> int

  (** Applies [f] to every leaf, keeping the structure. *)
  val map : 'a t -> (value -> value) -> 'a -> 'a

  (** Leafwise on two values of the same structure, e.g. [params - lr * grads].
      Raises [Type_error]. *)
  val map2 : 'a t -> (value -> value -> value) -> 'a -> 'a -> 'a
end
