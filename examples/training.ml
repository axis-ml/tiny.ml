(* Fit a small neural network to sin x with gradient descent. *)

open Tracer
open Common

(* [@@deriving tree] tells [Ad.grad] where the arrays are. *)
type layer =
  { w : value
  ; b : value
  }
[@@deriving tree]

type mlp =
  { hidden : layer
  ; out : layer
  }
[@@deriving tree]

let%ops dense { w; b } x = matmul x w +. b
let%ops predict net x = dense net.out (tanh (dense net.hidden x))
let%ops loss net (x, y) = mean (square (predict net x -. y))

let sgd_step data net =
  let grads = Ad.grad mlp_tree (fun net -> loss net data) net in
  Tree.map2 mlp_tree [%ops fun p g -> p -. (0.1 *. g)] net grads
;;

let init ~hidden =
  let rng = Random.State.make [| 42 |] in
  let random shape =
    let n = Array.fold_left ( * ) 1 shape in
    Ops.array ~shape (Array.init n (fun _ -> Random.State.float rng 2. -. 1.))
  in
  { hidden = { w = random [| 1; hidden |]; b = random [| hidden |] }
  ; out = { w = random [| hidden; 1 |]; b = Ops.zeros [| 1 |] }
  }
;;

let run () =
  let x = Ops.reshape ~shape:[| 32; 1 |] (Ops.linspace (-3.) 3. 32) in
  let data = x, Ops.sin x in
  (* Traced once, then replayed every iteration. *)
  let step = Compile.compile ~backend:Compile.jit mlp_tree mlp_tree (sgd_step data) in
  title "training";
  let net = ref (init ~hidden:16) in
  for i = 0 to 2000 do
    if i mod 400 = 0 then print (Printf.sprintf "loss at step %d" i) (loss !net data);
    net := step !net
  done;
  title "the network against sin";
  let probe = Ops.array ~shape:[| 4; 1 |] [| -2.; -0.5; 1.; 2.5 |] in
  print "x" (Ops.reshape ~shape:[| 4 |] probe);
  print "predict" (Ops.reshape ~shape:[| 4 |] (predict !net probe));
  print "sin x" (Ops.reshape ~shape:[| 4 |] (Ops.sin probe))
;;

let example =
  { name = "training"; summary = "fit an MLP to sin x with grad and jit"; run }
;;
