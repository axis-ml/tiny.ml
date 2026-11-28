(** "Raises [Type_error]" means raises [Error (Type_error _)]. *)
module Error : sig
  type t =
    | Type_error of string
    | Escaped_tracer of string
    | Concretization_error of string
    | Missing_rule of string

  exception Error of t

  (** e.g. ["Type_error: add: ..."]. *)
  val to_string : t -> string

  (** [Ok (f ())], or [Error e] if [f] raised [Error e]. Other exceptions pass
      through. *)
  val catch : (unit -> 'a) -> ('a, t) result

  (** Printf-style raisers, e.g. [type_error "add: got %d operands" n]. *)
  val type_error : ('a, unit, string, 'b) format4 -> 'a

  val escaped_tracer : ('a, unit, string, 'b) format4 -> 'a
  val concretization_error : ('a, unit, string, 'b) format4 -> 'a
  val missing_rule : ('a, unit, string, 'b) format4 -> 'a
end

(** On unless [TINY_ML_DEBUG] is unset, [""], ["0"] or ["false"]. *)
module Debug : sig
  val enabled : bool ref
end

(** Masks are [bool list]s as long as the list. *)
module Lists : sig
  (** [range 3] is [[0; 1; 2]]. *)
  val range : int -> int list

  (** First [n] elements and the rest. Raises [Invalid_argument] if too short. *)
  val split_at : int -> 'a list -> 'a list * 'a list

  (** Elements where the mask is [true]: [pick [true; false] [a; b]] is [[a]]. *)
  val pick : bool list -> 'a list -> 'a list

  (** Elements where the mask is [false]. *)
  val reject : bool list -> 'a list -> 'a list

  (** Inverse of [pick]/[reject]: [merge m (pick m l) (reject m l) = l]. *)
  val merge : bool list -> 'a list -> 'a list -> 'a list

  (** [Some xs] if every element is [Some], else [None]. *)
  val all_some : 'a option list -> 'a list option
end

(** Immutable maps keyed by var id. *)
module Env : sig
  type 'a t

  val empty : 'a t
  val add : int -> 'a -> 'a t -> 'a t

  (** Adds the pairs of two equal-length lists. *)
  val add_all : int list -> 'a list -> 'a t -> 'a t

  val find_opt : int -> 'a t -> 'a option
  val mem : int -> 'a t -> bool

  (** Like [Map.update]: [None] removes the key. *)
  val update : int -> ('a option -> 'a option) -> 'a t -> 'a t
end

(** Append-only log. *)
module Tape : sig
  type 'a t

  val create : unit -> 'a t

  (** Appends one entry. *)
  val record : 'a t -> 'a -> unit

  (** Oldest first. *)
  val to_list : 'a t -> 'a list

  (** Newest first. *)
  val find_map : ('a -> 'b option) -> 'a t -> 'b option
end

(** Dense ids and liveness, shared by [Ir] and [Linear]. *)
module Graph : sig
  (** Hands out [0], [1], [2], ... *)
  type counter

  val counter : unit -> counter

  (** Next unused id. *)
  val fresh : counter -> int

  (** How many ids were handed out. *)
  val count : counter -> int

  (** [nodes] in eval order; whether each reaches a root (dead-code check). *)
  val live
    :  inputs:('n -> int list)
    -> outputs:('n -> int list)
    -> roots:int list
    -> 'n list
    -> bool list
end

(** Helpers on shapes ([int array], outermost dim first). *)
module Shape : sig
  (** Number of elements: product of dims, [1] for [[||]]. *)
  val numel : int array -> int

  (** e.g. ["[2,3]"]. *)
  val to_string : int array -> string

  (** e.g. ["f64[2,3]"]. *)
  val describe : int array -> string

  (** C-layout strides in elements: [strides [|2; 3|]] is [[|3; 1|]]. *)
  val strides : int array -> int array

  (** Dims [0 .. rank-1] not in [axes], in order. *)
  val kept_dims : rank:int -> int list -> int list

  (** Drops dims [axes]: [remove_dims ~axes:[0] [|2; 3|]] is [[|3|]]. *)
  val remove_dims : axes:int list -> int array -> int array

  (** Inserts size-1 dims; [axes] are positions in the result. *)
  val insert_ones : axes:int list -> int array -> int array

  (** [s] with dim [axis] set to [n]. *)
  val set_dim : axis:int -> int -> int array -> int array

  (** Why [axes] is invalid for [rank] (out of range, duplicates), if it is. *)
  val axes_error : rank:int -> int list -> string option

  (** [perm] is a permutation of [0 .. rank-1]. *)
  val is_perm : rank:int -> int array -> bool

  (** [p'] with [p'.(p.(i)) = i]; undoes a transpose. *)
  val inverse_perm : int array -> int array

  (** Negative axes counted from the end, then sorted. *)
  val normalize_axes : rank:int -> int list -> int list

  (** NumPy broadcasting of two shapes; [None] if incompatible. *)
  val broadcast_shapes : int array -> int array -> int array option
end

(** Float64 C-layout arrays; never mutated. Errors raise [Invalid_argument]. *)
module Ndarray : sig
  type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

  (** [init shape f]: element [i] (flat index) is [f i]. *)
  val init : int array -> (int -> float) -> t

  (** Flat data in C order. *)
  val of_array : int array -> float array -> t

  val full : int array -> float -> t
  val zeros : int array -> t

  (** Rank 0. *)
  val scalar : float -> t

  (** [[0.; 1.; ...; n-1.]]. *)
  val arange : int -> t

  (** [n] evenly spaced points from [lo] to [hi], both included. *)
  val linspace : float -> float -> int -> t

  (** [n]x[n] identity. *)
  val eye : int -> t

  val shape : t -> int array
  val rank : t -> int

  (** Number of elements. *)
  val size : t -> int

  (** Flat data in C order. *)
  val to_array : t -> float array

  (** The value of a rank-0 array. *)
  val to_float : t -> float

  (** Elementwise; [map2] needs equal shapes (no broadcasting at this level). *)
  val map : (float -> float) -> t -> t

  val map2 : (float -> float -> float) -> t -> t -> t

  (** 1.0 where true, 0.0 elsewhere. *)
  val greater : t -> t -> t

  val less : t -> t -> t
  val maximum : t -> t -> t

  (** [select p a b]: [a] where [p > 0], else [b]; equal shapes. *)
  val select : t -> t -> t -> t

  (** Matrix product of [[m; k]] and [[k; n]]. *)
  val dot : t -> t -> t

  (** Sums away [axes]. *)
  val reduce_sum : axes:int list -> t -> t

  val reduce_max : axes:int list -> t -> t

  (** Same data, new shape (same [numel]). *)
  val reshape : shape:int array -> t -> t

  (** Inserts size-1 dims at [axes]. *)
  val expand_dims : axes:int list -> t -> t

  (** Same rank; repeats size-1 dims. *)
  val broadcast_to : shape:int array -> t -> t

  (** [np.broadcast_to (np.expand_dims a axes) shape]. *)
  val broadcast : shape:int array -> axes:int list -> t -> t

  (** Result dim [i] is input dim [perm.(i)]. *)
  val transpose : perm:int array -> t -> t

  (** Joins along an existing [axis]. *)
  val concatenate : axis:int -> t list -> t

  (** [start] to [stop] (exclusive) along [axis]. *)
  val slice : axis:int -> start:int -> stop:int -> t -> t

  (** Same shape and [|a - b| <= atol + rtol * |b|] everywhere. *)
  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool

  (** Shortest round-tripping repr. *)
  val float_repr : float -> string

  val shape_to_string : int array -> string

  (** Nested-bracket printing of the data. *)
  val to_string : t -> string

  val pp : Format.formatter -> t -> unit
end

(** Shape and dtype, no data: what a tracer knows about its value. *)
module Aval : sig
  type dtype = F64

  type t = private
    { shape : int array
    ; dtype : dtype
    }

  (** [dtype] defaults to [F64]. *)
  val create : ?dtype:dtype -> int array -> t

  val equal : t -> t -> bool
  val of_ndarray : Ndarray.t -> t

  (** e.g. ["f64"]. *)
  val dtype_name : dtype -> string

  (** e.g. ["f64[2,3]"]. *)
  val to_string : t -> string
end
