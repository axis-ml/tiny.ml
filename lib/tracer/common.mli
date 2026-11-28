(** "Raises [Type_error]" means raises [Error (Type_error _)]. *)
module Error : sig
  type t =
    | Type_error of string
    | Escaped_tracer of string
    | Concretization_error of string
    | Missing_rule of string

  exception Error of t

  val to_string : t -> string
  val catch : (unit -> 'a) -> ('a, t) result

  (** Printf-style raisers. *)
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
  val range : int -> int list

  (** Raises [Invalid_argument] if too short. *)
  val split_at : int -> 'a list -> 'a list * 'a list

  val pick : bool list -> 'a list -> 'a list
  val reject : bool list -> 'a list -> 'a list

  (** Inverse of [pick]/[reject]. *)
  val merge : bool list -> 'a list -> 'a list -> 'a list

  val all_some : 'a option list -> 'a list option
end

(** Immutable maps keyed by var id. *)
module Env : sig
  type 'a t

  val empty : 'a t
  val add : int -> 'a -> 'a t -> 'a t
  val add_all : int list -> 'a list -> 'a t -> 'a t
  val find_opt : int -> 'a t -> 'a option
  val mem : int -> 'a t -> bool
  val update : int -> ('a option -> 'a option) -> 'a t -> 'a t
end

(** Append-only log. *)
module Tape : sig
  type 'a t

  val create : unit -> 'a t
  val record : 'a t -> 'a -> unit

  (** Oldest first. *)
  val to_list : 'a t -> 'a list

  (** Newest first. *)
  val find_map : ('a -> 'b option) -> 'a t -> 'b option
end

(** Dense ids and liveness, shared by [Ir] and [Linear]. *)
module Graph : sig
  type counter

  val counter : unit -> counter
  val fresh : counter -> int
  val count : counter -> int

  (** [nodes] in eval order; whether each reaches a root. *)
  val live
    :  inputs:('n -> int list)
    -> outputs:('n -> int list)
    -> roots:int list
    -> 'n list
    -> bool list
end

module Shape : sig
  val numel : int array -> int
  val to_string : int array -> string
  val describe : int array -> string
  val strides : int array -> int array
  val kept_dims : rank:int -> int list -> int list
  val remove_dims : axes:int list -> int array -> int array

  (** [axes] are positions in the result. *)
  val insert_ones : axes:int list -> int array -> int array

  (** [s] with dim [axis] set to [n]. *)
  val set_dim : axis:int -> int -> int array -> int array

  val axes_error : rank:int -> int list -> string option
  val is_perm : rank:int -> int array -> bool
  val inverse_perm : int array -> int array
  val normalize_axes : rank:int -> int list -> int list
  val broadcast_shapes : int array -> int array -> int array option
end

(** Float64 C-layout arrays; never mutated. *)
module Ndarray : sig
  type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

  val init : int array -> (int -> float) -> t
  val of_array : int array -> float array -> t
  val full : int array -> float -> t
  val zeros : int array -> t
  val scalar : float -> t
  val arange : int -> t
  val linspace : float -> float -> int -> t
  val eye : int -> t
  val shape : t -> int array
  val rank : t -> int
  val size : t -> int
  val to_array : t -> float array
  val to_float : t -> float
  val map : (float -> float) -> t -> t
  val map2 : (float -> float -> float) -> t -> t -> t
  val greater : t -> t -> t
  val less : t -> t -> t
  val maximum : t -> t -> t
  val select : t -> t -> t -> t
  val dot : t -> t -> t
  val reduce_sum : axes:int list -> t -> t
  val reduce_max : axes:int list -> t -> t
  val reshape : shape:int array -> t -> t
  val expand_dims : axes:int list -> t -> t

  (** Same rank; repeats size-1 dims. *)
  val broadcast_to : shape:int array -> t -> t

  (** [np.broadcast_to (np.expand_dims a axes) shape]. *)
  val broadcast : shape:int array -> axes:int list -> t -> t

  val transpose : perm:int array -> t -> t
  val concatenate : axis:int -> t list -> t
  val slice : axis:int -> start:int -> stop:int -> t -> t
  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool

  (** Shortest round-tripping repr. *)
  val float_repr : float -> string

  val shape_to_string : int array -> string
  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

(** Shape and dtype, no data. *)
module Aval : sig
  type dtype = F64

  type t = private
    { shape : int array
    ; dtype : dtype
    }

  val create : ?dtype:dtype -> int array -> t
  val equal : t -> t -> bool
  val of_ndarray : Ndarray.t -> t
  val dtype_name : dtype -> string
  val to_string : t -> string
end
