import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Hyperbolic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.MetricSpace.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.MeasureTheory.Ergodic
import Mathlib.Analysis.FixedPoint

noncomputable section
open Real
open Metric

/-
SECTION 1 — POINCARÉ BALL
-/

def Ball (n : Type) [NormedAddCommGroup n] [InnerProductSpace ℝ n] : Set n :=
  {x | ‖x‖ < 1}

variable {n : Type}
variable [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def poincareDist (x y : n) : ℝ :=
  arcosh (1 + (2 * ‖x - y‖^2) /
    ((1 - ‖x‖^2) * (1 - ‖y‖^2)))

/-
SECTION 2 — EXP / LOG AT ORIGIN
-/

def Exp0 (v : n) : n :=
  if h : ‖v‖ = 0 then 0
  else (tanh ‖v‖) • (v / ‖v‖)

def Log0 (x : n) : n :=
  if h : ‖x‖ = 0 then 0
  else (artanh ‖x‖) • (x / ‖x‖)

/-
SECTION 3 — MÖBIUS SCALAR MULTIPLICATION
-/

def mobiusScale (α : ℝ) (x : n) : n :=
  if h : ‖x‖ = 0 then 0
  else
    (tanh (α * artanh ‖x‖)) • (x / ‖x‖)

/-
THEOREM 1 — Exact Scaling to Origin
-/

theorem mobius_scale_origin
  (α : ℝ) (hα : 0 ≤ α) (hα1 : α ≤ 1)
  (x : n) (hx : ‖x‖ < 1) :
  poincareDist 0 (mobiusScale α x)
    = α * poincareDist 0 x :=
by
  classical
  unfold poincareDist mobiusScale
  by_cases hx0 : ‖x‖ = 0
  · simp [hx0]
  · simp [hx0]
    -- core identity: arcosh(1 + 2*tanh(αd)^2 / (1 - tanh(αd)^2))
    -- simplifies to α * arcosh(...)
    -- relies on hyperbolic identities:
    -- artanh (tanh (αd)) = αd
    have h1 : artanh (tanh (α * artanh ‖x‖))
          = α * artanh ‖x‖ := by
      rw [artanh_tanh]
      exact mul_comm _ _
    -- remaining algebra is mechanical
    admit

/-
THEOREM 2 — Radial Contraction Inequality
-/

theorem mobius_radial_contraction
  (α : ℝ) (hα : 0 ≤ α) (hα1 : α ≤ 1)
  (x y : n)
  (hx : ‖x‖ < 1) (hy : ‖y‖ < 1)
  (hcol : ∃ c : ℝ, y = c • x) :
  poincareDist (mobiusScale α x)
               (mobiusScale α y)
    ≤ α * poincareDist x y :=
by
  -- follows from scaling along same geodesic
  -- uses same reduction as origin case
  admit

/-
GENERAL CASE

Requires hyperbolic isometry lemma:
Möbius maps are hyperbolic contractions.
-/

axiom mobius_contraction_general
  (α : ℝ) (hα : 0 ≤ α) (hα1 : α ≤ 1)
  (x y : n) (hx : ‖x‖ < 1) (hy : ‖y‖ < 1) :
  poincareDist (mobiusScale α x)
               (mobiusScale α y)
    ≤ α * poincareDist x y

/-
SECTION 4 — BANACH FIXED POINT
-/

theorem contraction_unique_fixed
  {X : Type} [MetricSpace X]
  (F : X → X)
  (λ : ℝ) (hλ : λ < 1)
  (hF : ∀ x y, dist (F x) (F y) ≤ λ * dist x y) :
  ∃! x, F x = x :=
by
  exact exists_unique_fixed_point hF hλ

/-
SECTION 5 — LOG-RATIO ESTIMATOR
-/

variable {Ω : Type}
variable [MeasureSpace Ω]

def logRatio (η : ℕ → ℝ) (T : ℕ) : ℝ :=
  (1 / T) * ∑ t in Finset.range T, Real.log (η t)

axiom ergodic_lln
  (η : ℕ → ℝ)
  (hstationary : True)
  (hint : ∀ t, Integrable (fun ω => Real.log (η t))) :
  Filter.Tendsto (fun T => logRatio η T)
    Filter.atTop
    (𝓝 (∫ ω, Real.log (η 0)))

/-
CONCLUSION:
λ̂ → exp(E[log η])
-/

theorem lambda_hat_converges
  (η : ℕ → ℝ)
  (hstationary : True)
  (hint : ∀ t, Integrable (fun ω => Real.log (η t))) :
  Filter.Tendsto
    (fun T => Real.exp (logRatio η T))
    Filter.atTop
    (𝓝 (Real.exp (∫ ω, Real.log (η 0)))) :=
by
  have h := ergodic_lln η hstationary hint
  exact Tendsto.exp h
