open Lang

(** Elementwise primitives in one call: registers type, impl, jvp and vmap
    rules. Reverse mode works if [deriv] is linear in the tangent. Raises
    [Invalid_argument] if [name] is taken.

    {[
      let softplus =
        Define.unary
          ~name:"softplus"
          ~impl:(fun x -> Float.log1p (Float.exp x))
          ~deriv:(fun x t -> Ops.(t * logistic x))
    ]} *)

type value := Interp.value

(** Returns the new op. [impl] runs per element; [deriv x t] is [t * f'(x)]. *)
val unary
  :  name:string
  -> impl:(float -> float)
  -> deriv:(value -> value -> value)
  -> value
  -> value

(** Returns the new op. Broadcasts like {!Ops.add}. [deriv x y] is
    [(tx -> tx * df/dx, ty -> ty * df/dy)]. *)
val binary
  :  name:string
  -> impl:(float -> float -> float)
  -> deriv:(value -> value -> (value -> value) * (value -> value))
  -> value
  -> value
  -> value
