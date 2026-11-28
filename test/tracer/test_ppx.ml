open Tracer

let s = Ops.scalar
let x = Ops.array [| 0.5; 1.; 2. |]
let%ops f x = -.(sin x *. 2.) +. x
let%ops h x = (x ** 4.) -. (3. *. x *. x) +. x
let%ops softplus x = log (1. +. exp x)
let%ops logistic x = 1. /. (1. +. exp (-.x))
let%ops relu x = x *. (x >. 0.)
let%ops masks x = (x <. 1.) +. (x >=. 1.) +. (x <=. 1.)
let%ops roots x = (x ** 0.5) +. sqrt x
let%ops pow_general x y = x ** y
let%ops float_funs x = Float.abs (-.x) +. abs_float x +. Float.max x 1. +. tanh x
let%ops hyperbolic x = sinh x +. cosh x +. tan x +. asinh (-.x) +. Float.tanh x
let%ops inverse x = acosh (x +. 1.) +. atanh (x /. 4.)
let%ops scaled x n = x *. float_of_int n
let%ops loss (w, b) x = mean (square (matmul w x +. b))

let%ops repeat_half x =
  let n = size x in
  reshape ~shape:[| n * 2 |] (concatenate [ x; x ]) *. 0.5
;;

let%ops branchy x = if to_float x > 0. then sin x else cos x

let () =
  Check.value "%ops: f" ~expected:Ops.(-(sin x * scalar 2.) + x) (f x);
  Check.value
    "%ops: ** 4. is integer_pow"
    ~expected:Ops.(integer_pow ~exponent:4 x - (scalar 3. * x * x) + x)
    (h x);
  Check.value
    "%ops: literals in function arguments"
    ~expected:Ops.(log (scalar 1. + exp x))
    (softplus x);
  Check.value
    "%ops: -. is neg"
    ~expected:Ops.(scalar 1. / (scalar 1. + exp (neg x)))
    (logistic x);
  Check.value
    "%ops: >. is a mask"
    ~expected:(Ops.array [| 0.; 2. |])
    (relu (Ops.array [| -1.; 2. |]));
  Check.value "%ops: <. >=. <=." ~expected:(Ops.array [| 2.; 2.; 1. |]) (masks x);
  Check.value "%ops: ** 0.5 is pow" ~expected:Ops.(scalar 2. * sqrt x) (roots x);
  Check.value
    ~rtol:1e-9
    "%ops: x ** y is exp (y log x)"
    ~expected:(s 8.)
    (pow_general (s 2.) (s 3.));
  Check.value
    "%ops: Float.abs, abs_float, Float.max, tanh"
    ~expected:Ops.(abs x + abs x + maximum x (scalar 1.) + tanh x)
    (float_funs x);
  let stdlib f = Array.map f [| 0.5; 1.; 2. |] |> Ops.array in
  Check.value
    ~rtol:1e-12
    "%ops: sinh cosh tan asinh tanh"
    ~expected:
      (stdlib (fun v ->
         Float.sinh v +. Float.cosh v +. Float.tan v +. Float.asinh (-.v) +. Float.tanh v))
    (hyperbolic x);
  Check.value
    ~rtol:1e-12
    "%ops: acosh atanh"
    ~expected:(stdlib (fun v -> Float.acosh (v +. 1.) +. Float.atanh (v /. 4.)))
    (inverse x);
  Check.value "%ops: float_of_int" ~expected:Ops.(x * scalar 3.) (scaled x 3);
  Check.value
    "%ops: int arithmetic is left alone"
    ~expected:(Ops.array [| 0.25; 0.5; 1.; 0.25; 0.5; 1. |])
    (repeat_half x);
  Check.value "%ops: inline" ~expected:Ops.(x * x) [%ops x *. x];
  let g =
    let%ops g x = 2. *. sin x in
    g
  in
  Check.value "%ops: let%ops ... in" ~expected:Ops.(scalar 2. * sin x) (g x)
;;

let () =
  Check.value
    "%ops under deriv"
    ~expected:Ops.(scalar 1. - (scalar 2. * cos x))
    (Ad.deriv f x);
  Check.value
    "%ops under grad: grad softplus = logistic"
    ~expected:(logistic x)
    (Ad.grad (fun x -> softplus x |> Ops.reduce_sum) x);
  Check.value
    "%ops: branch on a value under grad"
    ~expected:(Ops.cos (s 1.))
    (Ad.grad branchy (s 1.));
  Check.raises
    ~matches:Check.is_concretization
    "%ops: branch on a value while staging"
    (fun () -> Ir.make branchy (s 1.));
  let fast = Compile.compile ~backend:Compile.jit f in
  Check.value "%ops under jit" ~expected:(f x) (fast x)
;;

type layer =
  { w : value
  ; b : value
  }
[@@deriving tree]

type mlp =
  { hidden : layer list
  ; out : layer
  ; scale : value
  }
[@@deriving tree]

type 'p adam =
  { params : 'p
  ; m : 'p
  ; v : 'p
  }
[@@deriving tree]

type one = { only : value } [@@deriving tree]
type pairs = (value * value) array [@@deriving tree]

module M = struct
  type t = { a : value } [@@deriving tree]
end

type uses_m = { inner : M.t } [@@deriving tree]

let dense { w; b } x = Ops.(matmul w x + b)

let forward net x =
  let h = List.fold_left (fun x l -> dense l x |> Ops.relu) x net.hidden in
  Ops.(net.scale * dense net.out h)
;;

let () =
  let mat r c = Ops.full [| r; c |] 0.1
  and vec n = Ops.full [| n |] 0.1 in
  let net =
    { hidden = [ { w = mat 4 3; b = vec 4 }; { w = mat 4 4; b = vec 4 } ]
    ; out = { w = mat 1 4; b = vec 1 }
    ; scale = s 2.
    }
  in
  let leaves t v = Tree.leaves t v |> List.length in
  Check.bool "deriving: nested record leaves" (leaves mlp_tree net = 7);
  Check.bool
    "deriving: leaves in field order"
    (List.for_all2
       Ops.allclose
       (Tree.leaves layer_tree { w = s 1.; b = s 2. })
       [ s 1.; s 2. ]);
  Check.bool
    "deriving: type parameter"
    (leaves (adam_tree mlp_tree) { params = net; m = net; v = net } = 21);
  Check.bool "deriving: one field" (leaves one_tree { only = s 1. } = 1);
  Check.bool "deriving: alias" (leaves pairs_tree [| s 1., s 2.; s 3., s 4. |] = 4);
  Check.bool "deriving: M.t is M.tree" (leaves uses_m_tree { inner = { a = s 1. } } = 1);
  let rebuilt = Tree.map mlp_tree (Ops.mul (s 2.)) net in
  Check.value "deriving: map rebuilds the record" ~expected:(s 4.) rebuilt.scale;
  let x = Ops.array [| 1.; 2.; 3. |] in
  let loss net = forward net x |> Ops.reduce_sum in
  let manual =
    let open Tree in
    let layer =
      conv
        ~to_tree:(fun { w; b } -> w, b)
        ~of_tree:(fun (w, b) -> { w; b })
        (pair value value)
    in
    conv
      ~to_tree:(fun { hidden; out; scale } -> hidden, out, scale)
      ~of_tree:(fun (hidden, out, scale) -> { hidden; out; scale })
      (triple (list layer) layer value)
  in
  let g = Ad.grad_tree ~wrt:mlp_tree loss net in
  let g' = Ad.grad_tree ~wrt:manual loss net in
  Check.bool
    "deriving: grad matches a hand-written tree"
    (List.for_all2 Ops.allclose (Tree.leaves mlp_tree g) (Tree.leaves manual g'));
  let y, dy =
    Ad.jvp_tree
      ~input:[%tree: value * value list]
      ~output:[%tree: value]
      (fun (a, bs) -> List.fold_left Ops.mul a bs)
      (s 2., [ s 3.; s 4. ])
      (s 1., [ s 0.; s 0. ])
  in
  Check.value "%tree: primal" ~expected:(s 24.) y;
  Check.value "%tree: tangent" ~expected:(s 12.) dy
;;

let () = Check.finish ()
