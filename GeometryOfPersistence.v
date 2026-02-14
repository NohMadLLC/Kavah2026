Require Import Reals.
Require Import Lra.
Require Import Psatz.
Require Import Coquelicot.Coquelicot.
Require Import Coq.Vectors.Vector.

Import VectorNotations.

Open Scope R_scope.

(* ====================================================== *)
(* 1. Hyperbolic Functions                                *)
(* ====================================================== *)

Definition sinh (x : R) := (exp x - exp (-x)) / 2.
Definition cosh (x : R) := (exp x + exp (-x)) / 2.
Definition tanh (x : R) := sinh x / cosh x.

(* Logarithm-based inverse tanh *)
Definition atanh (x : R) := (/ 2) * ln ((1 + x) / (1 - x)).

Lemma cosh_pos : forall x, 0 < cosh x.
Proof.
  intros x. unfold cosh.
  assert (0 < exp x) by apply exp_pos.
  assert (0 < exp (-x)) by apply exp_pos.
  lra.
Qed.

Lemma cosh_nonzero : forall x, cosh x <> 0.
Proof.
  intros. apply Rgt_not_eq. apply cosh_pos.
Qed.

Lemma tanh_alt : forall x, tanh x = (exp (2 * x) - 1) / (exp (2 * x) + 1).
Proof.
  intros x. unfold tanh, sinh, cosh.
  field_simplify_eq; try (apply cosh_nonzero).
  replace (exp x - exp (- x)) with (exp (-x) * (exp (2*x) - 1)).
  - replace (exp x + exp (- x)) with (exp (-x) * (exp (2*x) + 1)).
    + field. split.
      * apply exp_neq_0.
      * assert (0 < exp (2*x)) by apply exp_pos. lra.
    + rewrite Rmult_plus_distr_l.
      rewrite <- exp_plus. replace (-x + 2*x) with x by lra.
      rewrite <- exp_plus. replace (-x + 0) with (-x) by lra.
      reflexivity.
  - rewrite Rmult_minus_distr_l.
    rewrite <- exp_plus. replace (-x + 2*x) with x by lra.
    rewrite <- exp_plus. replace (-x + 0) with (-x) by lra.
    reflexivity.
Qed.

Lemma tanh_bound : forall x, Rabs (tanh x) < 1.
Proof.
  intros x. rewrite tanh_alt.
  set (y := exp (2 * x)).
  assert (Hy : 0 < y) by (unfold y; apply exp_pos).
  
  apply Rabs_lt. split.
  - apply Rlt_div_l.
    + apply Rplus_lt_0_compat; lra.
    + lra.
  - apply Rlt_div_r.
    + apply Rplus_lt_0_compat; lra.
    + lra.
Qed.

(* ====================================================== *)
(* 2. Finite Dimensional Euclidean Space ℝⁿ               *)
(* ====================================================== *)

(* FIXED: Dependent pattern matching for vectors *)
Fixpoint dot {n} (v w : Vector.t R n) : R :=
  match v in Vector.t _ n return Vector.t R n -> R with
  | Vector.nil _ => fun _ => 0
  | Vector.cons _ x _ v' => fun w =>
      match w in Vector.t _ S_n return R with
      | Vector.cons _ y _ w' => x * y + dot v' w'
      | _ => 0 
      end
  | _ => fun _ => 0
  end w.

Definition norm {n} (v : Vector.t R n) := sqrt (dot v v).

Lemma dot_nonneg : forall n (v : Vector.t R n), 0 <= dot v v.
Proof.
  induction v.
  - simpl. lra.
  - simpl. set (d := dot v v). 
    assert (0 <= h * h) by apply Rle_0_sqr.
    lra.
Qed.

Lemma norm_nonneg : forall n (v : Vector.t R n), 0 <= norm v.
Proof.
  intros. unfold norm. apply sqrt_pos.
Qed.

(* ====================================================== *)
(* 3. Poincaré Ball Definition                            *)
(* ====================================================== *)

Definition PoincareBall (n : nat) := { v : Vector.t R n | norm v < 1 }.

(* ====================================================== *)
(* 4. Radial Hyperbolic Distance from Origin              *)
(* ====================================================== *)

Definition d_origin {n} (x : PoincareBall n) :=
  2 * atanh (norm (proj1_sig x)).

(* ====================================================== *)
(* 5. The Main Identity Proof                             *)
(* ====================================================== *)

Lemma atanh_tanh_identity : forall x, Rabs x < 1 -> tanh (atanh x) = x.
Proof.
  intros x Hx.
  rewrite tanh_alt.
  unfold atanh.
  
  (* Simplify the exponent: 2 * (1/2 * ln(...)) = ln(...) *)
  replace (2 * (/ 2 * ln ((1 + x) / (1 - x)))) 
    with (ln ((1 + x) / (1 - x))) by field.
  
  (* Prepare to remove ln *)
  assert (H_pos : 0 < (1 + x) / (1 - x)).
  {
    apply Rdiv_lt_0_compat.
    - apply Rabs_lt_between in Hx. lra.
    - apply Rabs_lt_between in Hx. lra.
  }
  
  rewrite exp_ln; try assumption.
  
  (* Algebraic simplification *)
  field.
  split.
  - apply Rgt_not_eq. apply Rabs_lt_between in Hx. lra.
  - (* Prove denominator is not zero *)
    replace (1 + x + (1 - x)) with 2 by ring.
    lra.
Qed.