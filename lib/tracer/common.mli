module Error : sig
  exception Type_error of string
  exception Escaped_tracer of string
  exception Missing_rule of string
  exception Concretization_error of string

  val type_error : ('a, unit, string, 'b) format4 -> 'a
end

(** On when [TINY_ML_DEBUG] is set to anything but [""], ["0"] or
    ["false"]. *)
module Debug : sig
  val enabled : bool ref
end

(** What [Ir] and the vjp tape share: dense ids and liveness. *)
module Graph : sig
  (** Hands out [0], [1], ... *)
  type counter

  val counter : unit -> counter
  val fresh : counter -> int

  (** How many ids were handed out; they are all [< count c]. *)
  val count : counter -> int

  (** [reaches ~num_ids ~inputs ~outputs ~roots nodes], with [nodes] newest
      first: for each node, whether one of its outputs reaches a root. *)
  val reaches
    :  num_ids:int
    -> inputs:('n -> int list)
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

  val axes_error : rank:int -> int list -> string option
  val is_perm : rank:int -> int array -> bool
  val normalize_axes : rank:int -> int list -> int list
  val broadcast_shapes : int array -> int array -> int array option
end

(** Reference float64 C-layout arrays. Functions never mutate their
    arguments. *)
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
  val reshape : t -> int array -> t
  val expand_dims : axes:int list -> t -> t

  (** Same rank; size-1 dims are repeated. *)
  val broadcast_to : int array -> t -> t

  (** [np.broadcast_to (np.expand_dims a axes) shape]. *)
  val broadcast : shape:int array -> axes:int list -> t -> t

  val transpose : perm:int array -> t -> t
  val concatenate : axis:int -> t list -> t
  val slice : axis:int -> start:int -> stop:int -> t -> t
  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool

  (** Shortest repr that reads back to the same float. *)
  val float_repr : float -> string

  val shape_to_string : int array -> string
  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

(** Abstract values: a shape and a dtype, no data. *)
module Aval : sig
  type dtype = F64

  type t =
    { shape : int array
    ; dtype : dtype
    }

  val create : ?dtype:dtype -> int array -> t
  val of_ndarray : Ndarray.t -> t
  val dtype_name : dtype -> string
  val to_string : t -> string
end
