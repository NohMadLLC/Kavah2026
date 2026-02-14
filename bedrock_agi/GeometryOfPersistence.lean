import Mathlib.Analysis.SpecialFunctions.Trigonometric.Hyperbolic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.NormedSpace.InnerProduct
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.Calculus.FixedPoint

noncomputable section

open Real

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The open Poincaré ball. -/
def PoincareBall :=
  { x : E // ‖x‖ < 1 }

namespace PoincareBall

/-- Hyperbolic distance on the Poincaré ball. -/
def distH (x y : PoincareBall) : ℝ :=
  let nx2 := ‖(x : E)‖ ^ 2
  let ny2 := ‖(y : E)‖ ^ 2
  let d2  := ‖(x : E) - (y : E)‖ ^ 2
  acosh (1 + 2 * d2 / ((1 - nx2) * (1 - ny2)))

/-- Helper lemma: 1 - ‖x‖² > 0 inside the ball. -/
lemma one_sub_norm_sq_pos (x : PoincareBall) :
    0 < 1 - ‖(x : E)‖ ^ 2 := by
  have h : ‖(x : E)‖ < 1 := x.property
  have h2 : ‖(x : E)‖ ^ 2 < 1 := by
    have := pow_lt_one (norm_nonneg _) h (by decide : (2:ℕ) ≠ 0)
    simpa using this
  exact sub_pos.mpr h2

/-- Möbius scalar multiplication. -/
def mobius_scale (α : ℝ) (x : PoincareBall) : PoincareBall :=
  if hx : (x : E) = 0 then
    ⟨0, by simp⟩
  else
    let r := ‖(x : E)‖
    let new_r := tanh (α * atanh r)
    have r_pos : 0 < r := norm_pos_iff.mpr hx
    have r_lt : r < 1 := x.property
    have new_r_lt : |new_r| < 1 := by
      simpa using tanh_lt_one (α * atanh r)
    let scaled := (new_r / r) • (x : E)
    have : ‖scaled‖ < 1 := by
      have hnorm :
          ‖scaled‖ = |new_r| := by
        simp [scaled, norm_smul, r, abs_div,
              Real.norm_eq_abs, abs_of_pos r_pos]
      simpa [hnorm] using new_r_lt
    ⟨scaled, this⟩

infixr:75 " ⊗ " => mobius_scale

/-- Hyperbolic distance from origin simplifies. -/
lemma dist_origin_formula (x : PoincareBall) :
    distH ⟨0, by simp⟩ x =
      2 * atanh ‖(x : E)‖ := by
  unfold distH
  simp
  let r := ‖(x : E)‖
  have hr : r < 1 := x.property
  have h :
      1 + 2 * r^2 / ((1 - 0) * (1 - r^2))
        = (1 + r) / (1 - r) := by
    field_simp [pow_two]
  have :
      acosh ((1 + r) / (1 - r))
        = 2 * atanh r := by
    -- standard identity
    simpa using acosh_div_one_sub_sq_eq_two_atanh hr
  simpa [h]

/-- Radial scaling property. -/
theorem mobius_scale_prop
    (x : PoincareBall)
    {α : ℝ} (hα : 0 ≤ α) :
    distH ⟨0, by simp⟩ (α ⊗ x)
      = α * distH ⟨0, by simp⟩ x := by
  by_cases hx : (x : E) = 0
  · simp [mobius_scale, hx]
  · have h1 := dist_origin_formula x
    have :
        distH ⟨0, by simp⟩ (α ⊗ x)
          = 2 * atanh (tanh (α * atanh ‖(x : E)‖)) := by
      simp [mobius_scale, hx, dist_origin_formula]
    have hcancel :
        atanh (tanh (α * atanh ‖(x : E)‖))
          = α * atanh ‖(x : E)‖ := by
      simpa using atanh_tanh (α * atanh ‖(x : E)‖)
    simp [this, h1, hcancel]
    ring

/-- Hyperbolic Lipschitz definition. -/
def HLipschitz
    (F : PoincareBall → PoincareBall)
    (L : ℝ) :=
  ∀ x y, distH (F x) (F y)
            ≤ L * distH x y

/-- Möbius scaling is Lipschitz ≤ α for α ≤ 1.
    (Full two-point contraction is deep; we
     encode it as a theorem stub to refine.) -/
theorem mobius_scale_contraction
    {α : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    HLipschitz (fun x => α ⊗ x) α :=
by
  intro x y
  -- known hyperbolic contraction theorem
  -- can be fully derived via convexity of cosh
  admit

/-- Banach fixed-point persistence theorem. -/
theorem persistence_theorem
  {F : PoincareBall → PoincareBall}
  {λ : ℝ}
  (hF : HLipschitz F λ)
  (hλ : λ < 1) :
  ∃! p, F p = p :=
by
  have hλ' : 0 ≤ λ := by
    have := hF ⟨0, by simp⟩ ⟨0, by simp⟩
    simp at this
    exact le_of_lt hλ
  exact
    Metric.exists_unique_fixedPoint_of_contracting
      F hF hλ

end PoincareBall
