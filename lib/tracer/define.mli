open Lang

(** Elementwise primitives in one call: registers type, impl, jvp and vmap
    rules. Reverse mode works if [deriv] is linear in the tangent. Raises
    [Invalid_argument] if [name] is taken. *)

type value := Interp.value

(** [deriv x t] is [t * f'(x)]. *)
val unary
  :  name:string
  -> impl:(float -> float)
  -> deriv:(value -> value -> value)
  -> value
  -> value

(** Broadcasts like {!Ops.add}. [deriv x y] is [(tx -> tx * df/dx, ty -> ty * df/dy)]. *)
val binary
  :  name:string
  -> impl:(float -> float -> float)
  -> deriv:(value -> value -> (value -> value) * (value -> value))
  -> value
  -> value
  -> value
