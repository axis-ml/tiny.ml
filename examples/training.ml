open Tracer
open Common

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
let%ops predict net x = x |> dense net.hidden |> tanh |> dense net.out
let%ops loss net (x, y) = predict net x -. y |> square |> mean

let sgd_step data net =
  let grads = Ad.grad_tree ~wrt:mlp_tree (fun net -> loss net data) net in
  Tree.map2 mlp_tree [%ops fun p g -> p -. (0.1 *. g)] net grads
;;

let init ~hidden =
  let rng = Random.State.make [| 42 |] in
  let random shape =
    let n = Array.fold_left ( * ) 1 shape in
    Array.init n (fun _ -> Random.State.float rng 2. -. 1.) |> Ops.array ~shape
  in
  { hidden = { w = random [| 1; hidden |]; b = random [| hidden |] }
  ; out = { w = random [| hidden; 1 |]; b = Ops.zeros [| 1 |] }
  }
;;

let log_loss ~loss i net = print (Printf.sprintf "loss at step %d" i) (loss net)

let train ~steps ~log_every ~step ~loss net0 =
  Seq.iterate step net0
  |> Seq.take (steps + 1)
  |> Seq.fold_lefti
       (fun _ i net ->
          if i < steps && i mod log_every = 0 then log_loss ~loss i net;
          net)
       net0
;;

let flat v = Ops.reshape ~shape:[| Ops.size v |] v

let run () =
  let x = Ops.linspace (-3.) 3. 32 |> Ops.reshape ~shape:[| 32; 1 |] in
  let data = x, Ops.sin x in
  let step =
    Compile.compile_tree
      ~backend:Compile.jit
      ~input:mlp_tree
      ~output:mlp_tree
      (sgd_step data)
  in
  title "training";
  let net =
    train
      ~steps:2001
      ~log_every:400
      ~step
      ~loss:(fun net -> loss net data)
      (init ~hidden:16)
  in
  title "the network against sin";
  let probe = Ops.array ~shape:[| 4; 1 |] [| -2.; -0.5; 1.; 2.5 |] in
  flat probe |> print "x";
  predict net probe |> flat |> print "predict";
  Ops.sin probe |> flat |> print "sin x"
;;

let example =
  { name = "training"; summary = "fit an MLP to sin x with grad and jit"; run }
;;
