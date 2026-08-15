from manim import *
import numpy as np
import math

# ── colour palette ──────────────────────────────────────────────────────────
C_BG        = "#0d1117"
C_ACCENT    = "#58a6ff"
C_GREEN     = "#3fb950"
C_RED       = "#f85149"
C_PURPLE    = "#bc8cff"
C_YELLOW    = "#e3b341"
C_WHITE     = "#e6edf3"
C_GREY      = "#8b949e"

config.background_color = C_BG


# ════════════════════════════════════════════════════════════════════════════
# 1 – TITLE CARD
# ════════════════════════════════════════════════════════════════════════════
class S01_Title(Scene):
    def construct(self):
        conf = Text("IEEE CASE 2026 · Paper #454", font_size=20, color=C_GREY)
        conf.to_edge(UP, buff=0.4)

        title = Text(
            "Robust Single-Pass\nUncertainty Estimation",
            font_size=52, color=C_WHITE, weight=BOLD, line_spacing=1.2
        ).center()

        sub = Text(
            "for Resource-Constrained DNNs in the Presence of Outliers",
            font_size=22, color=C_ACCENT
        ).next_to(title, DOWN, buff=0.4)

        bar = Line(LEFT * 5, RIGHT * 5, color=C_ACCENT, stroke_width=1.5)
        bar.next_to(sub, DOWN, buff=0.15)

        tag = Text(
            "Generalised Normal Loss  ·  Automatic Shape Adaptation",
            font_size=17, color=C_GREY
        ).next_to(bar, DOWN, buff=0.15)

        self.play(FadeIn(conf, shift=DOWN * 0.3))
        self.play(Write(title, run_time=2))
        self.play(FadeIn(bar), Write(sub, run_time=1.5))
        self.play(FadeIn(tag))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 2 – TOY REGRESSION PROBLEM
# ════════════════════════════════════════════════════════════════════════════
class S02_ToyRegression(Scene):
    def construct(self):
        title = Text("Toy Regression Problem",
                     font_size=28, color=C_ACCENT, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(
            x_range=[-0.5, 5.5, 1], y_range=[-1, 6, 1],
            x_length=8, y_length=4.5,
            axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        xl = ax.get_x_axis_label(Text("x", font_size=22, color=C_GREY))
        yl = ax.get_y_axis_label(Text("y", font_size=22, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35

        dots = VGroup(*[
            Dot(ax.coords_to_point(x, y), radius=0.09, color=C_ACCENT, fill_opacity=0.9)
            for x, y in zip(xs, ys)
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.08, run_time=1.4))

        true_line = ax.plot(lambda x: x, x_range=[0, 5.2], color=C_GREEN, stroke_width=2.5)
        lbl_true = Text("True: y = x + noise", font_size=17, color=C_GREEN)
        lbl_true.next_to(ax.coords_to_point(3.5, 5.0), RIGHT, buff=0.05)
        self.play(Create(true_line), Write(lbl_true))

        cap = Text(
            "Goal: learn mean μ(x) and uncertainty σ(x) from noisy data",
            font_size=17, color=C_WHITE
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(cap))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 3 – GAUSSIAN NLL LOSS + TRAINING ANIMATION
# ════════════════════════════════════════════════════════════════════════════
class S03_GaussianLoss(Scene):
    def construct(self):
        title = Text("Gaussian NLL Loss & Training",
                     font_size=28, color=C_ACCENT, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        formula = MathTex(
            r"\mathcal{L}_{\text{Gauss}} = \frac{r^2}{2\alpha^2} + \log(\alpha)",
            font_size=38, color=C_WHITE
        ).shift(UP * 2.0)
        self.play(Write(formula, run_time=1.5))

        ax = Axes(
            x_range=[-3, 3, 1], y_range=[0, 5, 1],
            x_length=7, y_length=3.5,
            axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.9)
        xl = ax.get_x_axis_label(Text("residual  r", font_size=18, color=C_GREY))
        yl = ax.get_y_axis_label(Text("loss", font_size=18, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        alpha = 1.0
        gauss_curve = ax.plot(
            lambda r: r**2 / (2 * alpha**2) + 2 * np.log(alpha),
            x_range=[-2.8, 2.8], color=C_GREEN, stroke_width=2.5
        )
        lbl_g = Text("Gaussian NLL  (β=2)", font_size=16, color=C_GREEN)
        lbl_g.next_to(ax.coords_to_point(1.5, 4.5), RIGHT, buff=0.05)
        self.play(Create(gauss_curve), Write(lbl_g))

        # Ball rolling to minimum
        r_tracker = ValueTracker(2.5)
        ball = Dot(color=C_YELLOW, radius=0.13)
        ball.add_updater(lambda m: m.move_to(
            ax.coords_to_point(r_tracker.get_value(),
                               r_tracker.get_value()**2/(2*alpha**2) + 2*np.log(alpha))
        ))
        self.add(ball)
        trace = TracedPath(ball.get_center, stroke_color=C_YELLOW,
                           stroke_width=1.5, stroke_opacity=0.6)
        self.add(trace)

        self.play(r_tracker.animate.set_value(0.0), run_time=2.5,
                  rate_func=rate_functions.ease_out_cubic)
        ball.clear_updaters()

        lbl_min = Text("minimum  r=0", font_size=15, color=C_YELLOW)
        lbl_min.next_to(ball, UR, buff=0.15)
        self.play(Write(lbl_min))

        res_note = Text(
            "Gradient pushes each residual toward 0 → model learns the data distribution",
            font_size=16, color=C_WHITE
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(res_note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 4 – GAUSSIAN UNCERTAINTY BAND (clean)
# ════════════════════════════════════════════════════════════════════════════
class S04_GaussianUncertainty(Scene):
    def construct(self):
        title = Text("Gaussian Uncertainty on Clean Data",
                     font_size=26, color=C_GREEN, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(
            x_range=[-0.3, 5.3, 1], y_range=[-1, 7, 1],
            x_length=8.5, y_length=4.8,
            axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        self.play(Create(ax))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35
        dots = VGroup(*[Dot(ax.coords_to_point(x, y), radius=0.09, color=C_ACCENT)
                        for x, y in zip(xs, ys)])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.08, run_time=1.2))

        mu_line = ax.plot(lambda x: x, x_range=[0, 5.1], color=C_GREEN, stroke_width=2.5)
        self.play(Create(mu_line))

        sigma = 0.4
        upper = ax.plot(lambda x: x + 2*sigma, x_range=[0, 5.1], color=C_GREEN, stroke_width=0)
        lower = ax.plot(lambda x: x - 2*sigma, x_range=[0, 5.1], color=C_GREEN, stroke_width=0)
        band  = ax.get_area(upper, bounded_graph=lower, color=C_GREEN, opacity=0.25)
        self.play(FadeIn(band))

        lbl_band = Text("95% prediction interval  (±2σ)", font_size=17, color=C_GREEN)
        lbl_band.next_to(ax.coords_to_point(3.5, 5.5), RIGHT, buff=0.05)
        self.play(Write(lbl_band))

        cap = Text("Clean data → tight, well-calibrated uncertainty band",
                   font_size=17, color=C_WHITE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(cap))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 5 – OUTLIER CAUSES HIGH GAUSSIAN LOSS
# ════════════════════════════════════════════════════════════════════════════
class S05_OutlierGaussian(Scene):
    def construct(self):

        # ── PART A: regression scatter with the outlier ──────────────────────
        title = Text("Data with Outlier",
                     font_size=26, color=C_RED, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax_reg = Axes(
            x_range=[-0.3, 5.3, 1], y_range=[-1, 9, 1],
            x_length=8.5, y_length=4.8,
            axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        xl_r = ax_reg.get_x_axis_label(Text("x", font_size=20, color=C_GREY))
        yl_r = ax_reg.get_y_axis_label(Text("y", font_size=20, color=C_GREY))
        self.play(Create(ax_reg), Write(xl_r), Write(yl_r))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35

        clean_dots = VGroup(*[
            Dot(ax_reg.coords_to_point(x, y), radius=0.09, color=C_ACCENT)
            for x, y in zip(xs, ys)
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in clean_dots],
                              lag_ratio=0.08, run_time=1.2))

        # fitted line on clean data
        mu_line = ax_reg.plot(lambda x: x, x_range=[0, 5.1],
                              color=C_GREEN, stroke_width=2.5)
        self.play(Create(mu_line))

        # now drop the outlier
        outlier_reg = Dot(ax_reg.coords_to_point(2.5, 7.8),
                          radius=0.14, color=C_RED)
        outlier_reg_lbl = Text("Outlier!", font_size=17, color=C_RED, weight=BOLD)
        outlier_reg_lbl.next_to(outlier_reg, UR, buff=0.12)

        # residual arrow from fitted line to outlier
        fitted_y_at_x = 2.5          # μ(2.5) = 2.5
        res_arrow = Arrow(
            ax_reg.coords_to_point(2.5, fitted_y_at_x),
            ax_reg.coords_to_point(2.5, 7.8),
            color=C_RED, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.06
        )
        res_lbl = Text("large\nresidual r", font_size=14, color=C_RED,
                        line_spacing=1.1)
        res_lbl.next_to(res_arrow, RIGHT, buff=0.1)

        self.play(GrowFromCenter(outlier_reg), Write(outlier_reg_lbl))
        self.play(GrowArrow(res_arrow), Write(res_lbl))

        cap_a = Text("One outlier → a very large residual  r = y − μ(x)",
                     font_size=16, color=C_WHITE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(cap_a))
        self.wait(2.0)

        # ── CLEAR everything, keep only title ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── PART B: loss-function view ────────────────────────────────────────
        title2 = Text("Outlier → Huge Gaussian Loss",
                      font_size=26, color=C_RED, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title2))

        ax = Axes(
            x_range=[-3.2, 3.2, 1], y_range=[0, 7, 1],
            x_length=7.5, y_length=4.2,
            axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.5)
        xl = ax.get_x_axis_label(Text("residual  r", font_size=18, color=C_GREY))
        yl = ax.get_y_axis_label(Text("loss", font_size=18, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        alpha = 1.0
        gauss_curve = ax.plot(
            lambda r: r**2/(2*alpha**2) + 2*np.log(alpha),
            x_range=[-3.0, 3.0], color=C_GREEN, stroke_width=2.5
        )
        lbl_g = Text("Gaussian NLL", font_size=16, color=C_GREEN)
        lbl_g.next_to(ax.coords_to_point(1.8, 5.0), RIGHT, buff=0.05)
        self.play(Create(gauss_curve), Write(lbl_g))

        # normal residual dots sitting near r = 0
        normal_dots = VGroup(*[
            Dot(ax.coords_to_point(r, r**2/(2*alpha**2)),
                radius=0.09, color=C_ACCENT)
            for r in [-0.6, -0.3, 0.0, 0.3, 0.7]
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in normal_dots],
                              lag_ratio=0.12, run_time=1.0))

        # outlier residual far out on the curve
        r_out   = 2.8
        loss_out = r_out**2/(2*alpha**2)
        outlier_dot = Dot(ax.coords_to_point(r_out, loss_out),
                          radius=0.14, color=C_RED)
        outlier_lbl = Text("Outlier residual", font_size=15,
                            color=C_RED, weight=BOLD)
        outlier_lbl.next_to(outlier_dot, UR, buff=0.12)

        v_arrow = Arrow(
            ax.coords_to_point(r_out, 0),
            ax.coords_to_point(r_out, loss_out),
            color=C_RED, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.08
        )
        loss_lbl = Text("Very high loss!", font_size=15, color=C_RED)
        loss_lbl.next_to(v_arrow, RIGHT, buff=0.1)

        self.play(GrowFromCenter(outlier_dot), Write(outlier_lbl))
        self.play(GrowArrow(v_arrow), Write(loss_lbl))

        grad_note = Text(
            "Large gradient  →  model mean and σ pulled toward the outlier",
            font_size=16, color=C_RED
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(grad_note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


#         title = Text("Outlier → Huge Gaussian Loss",
#                      font_size=26, color=C_RED, weight=BOLD).to_edge(UP, buff=0.35)
#         self.play(Write(title))

#         ax = Axes(
#             x_range=[-0.3, 5.3, 1], y_range=[-2, 9, 1],
#             x_length=8.5, y_length=4.8,
#             axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
#         ).shift(DOWN * 0.3)
#         self.play(Create(ax))

#         np.random.seed(42)
#         xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
#         ys = xs + np.random.randn(len(xs)) * 0.35

#         clean_dots = VGroup(*[
#             Dot(ax.coords_to_point(x, y), radius=0.09, color=C_ACCENT)
#             for x, y in zip(xs, ys)
#         ])
#         self.play(LaggedStart(*[GrowFromCenter(d) for d in clean_dots],
#                               lag_ratio=0.07, run_time=1.0))

#         outlier_dot = Dot(ax.coords_to_point(2.5, 7.5), radius=0.13, color=C_RED)
#         out_lbl = Text("Outlier", font_size=16, color=C_RED)
#         out_lbl.next_to(outlier_dot, UP, buff=0.12)
#         self.play(GrowFromCenter(outlier_dot), Write(out_lbl))

#         ax = Axes(
#             x_range=[-3.2, 3.2, 1], y_range=[0, 7, 1],
#             x_length=7.5, y_length=4.2,
#             axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
#         ).shift(DOWN * 0.5)
#         xl = ax.get_x_axis_label(Text("residual  r", font_size=18, color=C_GREY))
#         yl = ax.get_y_axis_label(Text("loss", font_size=18, color=C_GREY))
#         self.play(Create(ax), Write(xl), Write(yl))

#         alpha = 1.0
#         gauss_curve = ax.plot(
#             lambda r: r**2/(2*alpha**2) + 2*np.log(alpha),
#             x_range=[-3.0, 3.0], color=C_GREEN, stroke_width=2.5
#         )
#         lbl_g = Text("Gaussian NLL", font_size=16, color=C_GREEN)
#         lbl_g.next_to(ax.coords_to_point(1.8, 5.0), RIGHT, buff=0.05)
#         self.play(Create(gauss_curve), Write(lbl_g))

#         normal_dots = VGroup(*[
#             Dot(ax.coords_to_point(r, r**2/(2*alpha**2)),
#                 radius=0.09, color=C_ACCENT)
#             for r in [-0.6, -0.3, 0.0, 0.3, 0.7]
#         ])
#         self.play(LaggedStart(*[GrowFromCenter(d) for d in normal_dots],
#                               lag_ratio=0.12, run_time=1.0))

#         r_out = 2.8
#         loss_out = r_out**2/(2*alpha**2)
#         outlier_dot = Dot(ax.coords_to_point(r_out, loss_out), radius=0.14, color=C_RED)
#         outlier_lbl = Text("Outlier!", font_size=17, color=C_RED, weight=BOLD)
#         outlier_lbl.next_to(outlier_dot, UR, buff=0.12)

#         v_arrow = Arrow(
#             ax.coords_to_point(r_out, 0),
#             ax.coords_to_point(r_out, loss_out),
#             color=C_RED, stroke_width=2, buff=0,
#             max_tip_length_to_length_ratio=0.08
#         )
#         loss_lbl = Text("Very high loss!", font_size=15, color=C_RED)
#         loss_lbl.next_to(v_arrow, RIGHT, buff=0.1)

#         self.play(GrowFromCenter(outlier_dot), Write(outlier_lbl))
#         self.play(GrowArrow(v_arrow), Write(loss_lbl))

#         grad_note = Text(
#             "Large gradient  →  model mean and σ pulled toward the outlier",
#             font_size=16, color=C_RED
#         ).to_edge(DOWN, buff=0.3)
#         self.play(Write(grad_note))
#         self.wait(2.5)
#         self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 6 – GAUSSIAN UNCERTAINTY DISTORTED BY OUTLIER
# ════════════════════════════════════════════════════════════════════════════
class S06_GaussianWithOutlier(Scene):
    def construct(self):
        title = Text("Gaussian Uncertainty Distorted by Outlier",
                     font_size=24, color=C_RED, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(
            x_range=[-0.3, 5.3, 1], y_range=[-2, 9, 1],
            x_length=8.5, y_length=4.8,
            axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        self.play(Create(ax))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35

        clean_dots = VGroup(*[
            Dot(ax.coords_to_point(x, y), radius=0.09, color=C_ACCENT)
            for x, y in zip(xs, ys)
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in clean_dots],
                              lag_ratio=0.07, run_time=1.0))

        outlier_dot = Dot(ax.coords_to_point(2.5, 7.5), radius=0.13, color=C_RED)
        out_lbl = Text("Outlier", font_size=16, color=C_RED)
        out_lbl.next_to(outlier_dot, UP, buff=0.12)
        self.play(GrowFromCenter(outlier_dot), Write(out_lbl))

        mu_line = ax.plot(lambda x: x + 0.5, x_range=[0, 5.1],
                          color=C_GREEN, stroke_width=2.5)
        self.play(Create(mu_line))

        sigma_inflated = 1.4
        upper2 = ax.plot(lambda x: x + 0.5 + 2*sigma_inflated,
                         x_range=[0, 5.1], color=C_RED, stroke_width=0)
        lower2 = ax.plot(lambda x: x + 0.5 - 2*sigma_inflated,
                         x_range=[0, 5.1], color=C_RED, stroke_width=0)
        band2 = ax.get_area(upper2, bounded_graph=lower2, color=C_RED, opacity=0.20)
        self.play(FadeIn(band2))

        warn = Text("Wide inflated band  →  under-confident, unreliable predictions",
                    font_size=17, color=C_RED).to_edge(DOWN, buff=0.3)
        self.play(Write(warn))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 7 – GENERALISED NORMAL: SHAPE PARAMETER β
# ════════════════════════════════════════════════════════════════════════════
class S07_GeneralizedShape(Scene):
    def construct(self):
        title = Text("Generalised Normal Loss – Shape Parameter β",
                     font_size=24, color=C_PURPLE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        formula = MathTex(
            r"\mathcal{L}_{\text{gen}} = \left(\frac{|r|}{\alpha}\right)^{\!\beta}"
            r"- \log\beta + \log\!\left(2\alpha\,\Gamma\!\left(\tfrac{1}{\beta}\right)\right)",
            font_size=30, color=C_WHITE
        ).shift(UP * 2.2)
        self.play(Write(formula, run_time=2))

        ax = Axes(
            x_range=[-3.2, 3.2, 1], y_range=[0, 5.5, 1],
            x_length=7.5, y_length=3.6,
            axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.9)
        xl = ax.get_x_axis_label(Text("residual  r", font_size=18, color=C_GREY))
        yl = ax.get_y_axis_label(Text("loss", font_size=18, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        alpha = 1.0

        def gen_nll(r, beta):
            return (abs(r)/alpha)**beta - np.log(beta) + np.log(
                2 * alpha * math.gamma(1/beta)
            )

        configs = [
            (2.0, C_GREEN,  "β=2  (Gaussian)"),
            (1.0, C_ACCENT, "β=1  (Laplace)"),
            (0.5, C_PURPLE, "β=0.5  (near-Cauchy)"),
        ]

        for beta, col, name in configs:
            try:
                curve = ax.plot(
                    lambda r, b=beta: gen_nll(r, b),
                    x_range=[-2.9, 2.9], color=col, stroke_width=2.5
                )
                lbl = Text(name, font_size=15, color=col)
                lbl.next_to(ax.coords_to_point(2.0, gen_nll(2.0, beta)), RIGHT, buff=0.05)
                self.play(Create(curve), Write(lbl), run_time=0.9)
            except Exception:
                pass

        flat_note = Text(
            "Smaller β  →  heavier tails  →  outliers produce smaller gradients",
            font_size=16, color=C_YELLOW
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(flat_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 8 – FLAT TAIL: OUTLIER HAS SMALL GRADIENT
# ════════════════════════════════════════════════════════════════════════════
class S08_FlatTail(Scene):
    def construct(self):
        title = Text("Outlier on Flat Tail  →  Near-Zero Gradient",
                     font_size=24, color=C_PURPLE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(
            x_range=[-3.2, 3.2, 1], y_range=[0, 5.5, 1],
            x_length=7.5, y_length=4.0,
            axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
        ).shift(DOWN * 0.7)
        xl = ax.get_x_axis_label(Text("residual  r", font_size=18, color=C_GREY))
        yl = ax.get_y_axis_label(Text("loss", font_size=18, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        alpha  = 1.0
        beta_g = 2.0
        beta_v = 0.8

        gauss_curve = ax.plot(
            lambda r: r**2/(2*alpha**2) + 2*np.log(alpha),
            x_range=[-3.0, 3.0], color=C_GREEN, stroke_width=2.5
        )
        gen_curve = ax.plot(
            lambda r: (abs(r)/alpha)**beta_v - np.log(beta_v) + np.log(2*alpha*math.gamma(1/beta_v)),
            x_range=[-3.0, 3.0], color=C_PURPLE, stroke_width=2.5
        )
        lbl_g   = Text("Gaussian (β=2)",       font_size=15, color=C_GREEN)
        lbl_gen = Text("Generalised (β=0.8)",  font_size=15, color=C_PURPLE)
        lbl_g.next_to(ax.coords_to_point(1.4, 4.8), RIGHT, buff=0.05)
        lbl_gen.next_to(ax.coords_to_point(-2.8, 3.5), LEFT, buff=0.05)

        self.play(Create(gauss_curve), Write(lbl_g))
        self.play(Create(gen_curve),   Write(lbl_gen))

        r_out = 2.6

        # Gaussian dot + steep tangent
        g_y    = r_out**2/(2*alpha**2) + 2*np.log(alpha)
        g_dot  = Dot(ax.coords_to_point(r_out, g_y), radius=0.12, color=C_RED)
        g_slp  = r_out / alpha**2
        dx = 0.5
        g_tang = Line(
            ax.coords_to_point(r_out-dx, g_y - g_slp*dx),
            ax.coords_to_point(r_out+dx, g_y + g_slp*dx),
            color=C_RED, stroke_width=2.5
        )
        g_grad = Text(f"∂L/∂r ≈ {g_slp:.1f}  (STEEP)", font_size=14, color=C_RED)
        g_grad.next_to(g_tang, UR, buff=0.1)

        # Generalised dot + flat tangent
        gen_y   = (abs(r_out)/alpha)**beta_v - np.log(beta_v) + np.log(2*alpha*math.gamma(1/beta_v))
        gen_dot = Dot(ax.coords_to_point(r_out, gen_y), radius=0.12, color=C_PURPLE)
        gen_slp = beta_v * r_out**(beta_v-1) / alpha**beta_v
        gen_tang = Line(
            ax.coords_to_point(r_out-dx, gen_y - gen_slp*dx),
            ax.coords_to_point(r_out+dx, gen_y + gen_slp*dx),
            color=C_PURPLE, stroke_width=2.5
        )
        gen_grad = Text(f"∂L/∂r ≈ {gen_slp:.2f}  (flat)", font_size=14, color=C_PURPLE)
        gen_grad.next_to(gen_tang, DR, buff=0.1)

        self.play(GrowFromCenter(g_dot), Create(g_tang), Write(g_grad))
        self.play(GrowFromCenter(gen_dot), Create(gen_tang), Write(gen_grad))

        note = Text(
            "Generalised: outlier gradient ≈ 0  →  model simply ignores the outlier",
            font_size=16, color=C_YELLOW
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 9 – AUTOMATIC β ADAPTATION
# ════════════════════════════════════════════════════════════════════════════
class S09_BetaAdaptation(Scene):
    def construct(self):
        title = Text("Automatic β Adaptation During Training",
                     font_size=24, color=C_PURPLE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        arrow = Arrow(LEFT * 4.2, RIGHT * 4.2, color=C_WHITE, stroke_width=2).shift(UP * 0.7)
        sl = Text("β = 2\n(Gaussian)", font_size=17, color=C_GREEN, line_spacing=1.1)
        sl.next_to(arrow, LEFT, buff=0.2).shift(UP * 0.2)
        el = Text("β → optimal\n(Laplace / Cauchy)", font_size=17, color=C_PURPLE, line_spacing=1.1)
        el.next_to(arrow, RIGHT, buff=0.2).shift(UP * 0.2)
        ml = Text("gradient-based optimisation of β", font_size=16, color=C_GREY)
        ml.next_to(arrow, DOWN, buff=0.2)

        self.play(GrowArrow(arrow))
        self.play(Write(sl), Write(el), Write(ml))

        ax = Axes(
            x_range=[0, 50, 10], y_range=[0.8, 2.2, 0.5],
            x_length=7, y_length=2.5,
            axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.5},
        ).shift(DOWN * 1.9)
        xl = ax.get_x_axis_label(Text("epoch", font_size=16, color=C_GREY))
        yl = ax.get_y_axis_label(Text("β", font_size=16, color=C_GREY))
        self.play(Create(ax), Write(xl), Write(yl))

        beta_path = ax.plot(
            lambda e: 2.0 * np.exp(-0.04*e) + 0.9*(1 - np.exp(-0.04*e)),
            x_range=[0, 50], color=C_PURPLE, stroke_width=2.5
        )
        self.play(Create(beta_path, run_time=2.5))

        note = Text(
            "β decreases automatically  →  heavier tails  →  robust to outliers",
            font_size=16, color=C_WHITE
        ).to_edge(DOWN, buff=0.25)
        self.play(Write(note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 10 – GENERALISED UNCERTAINTY (tight, correct)
# ════════════════════════════════════════════════════════════════════════════
class S10_GeneralizedUncertainty(Scene):
    def construct(self):
        title = Text("Generalised Uncertainty – Outlier Ignored",
                     font_size=24, color=C_PURPLE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(
            x_range=[-0.3, 5.3, 1], y_range=[-2, 9, 1],
            x_length=8.5, y_length=4.8,
            axis_config={"color": C_GREY, "include_tip": True, "stroke_width": 1.5},
        ).shift(DOWN * 0.3)
        self.play(Create(ax))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35
        dots = VGroup(*[Dot(ax.coords_to_point(x, y), radius=0.09, color=C_ACCENT)
                        for x, y in zip(xs, ys)])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.07, run_time=1.0))

        outlier_dot = Dot(ax.coords_to_point(2.5, 7.5), radius=0.13,
                          color=C_RED, fill_opacity=0.4)
        out_lbl = Text("Outlier\n(ignored)", font_size=14, color=C_RED, line_spacing=1.1)
        out_lbl.next_to(outlier_dot, UP, buff=0.1)
        out_lbl.set_opacity(0.5)
        self.play(GrowFromCenter(outlier_dot), Write(out_lbl))

        mu_line = ax.plot(lambda x: x, x_range=[0, 5.1],
                          color=C_PURPLE, stroke_width=2.5)
        self.play(Create(mu_line))

        sigma = 0.45
        upper = ax.plot(lambda x: x + 2*sigma, x_range=[0, 5.1], color=C_PURPLE, stroke_width=0)
        lower = ax.plot(lambda x: x - 2*sigma, x_range=[0, 5.1], color=C_PURPLE, stroke_width=0)
        band  = ax.get_area(upper, bounded_graph=lower, color=C_PURPLE, opacity=0.25)
        self.play(FadeIn(band))

        lbl_band = Text("Tight ±2σ  →  well-calibrated", font_size=17, color=C_PURPLE)
        lbl_band.next_to(ax.coords_to_point(3.5, 5.5), RIGHT, buff=0.05)
        self.play(Write(lbl_band))

        cap = Text(
            "Generalised loss: outlier near-zero gradient → mean and σ remain correct",
            font_size=16, color=C_WHITE
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(cap))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 11 – SIDE-BY-SIDE COMPARISON
# ════════════════════════════════════════════════════════════════════════════
class S11_Comparison(Scene):
    def construct(self):
        title = Text("Side-by-Side Comparison",
                     font_size=28, color=C_ACCENT, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        np.random.seed(42)
        xs = np.array([0.4, 0.9, 1.5, 2.0, 2.6, 3.1, 3.7, 4.3, 4.9])
        ys = xs + np.random.randn(len(xs)) * 0.35

        def make_panel(shift, colour, band_w, mu_off, label_str):
            ax = Axes(
                x_range=[-0.2, 5.2, 1], y_range=[-2.5, 9, 1],
                x_length=4.3, y_length=4.0,
                axis_config={"color": C_GREY, "include_tip": False, "stroke_width": 1.2},
            ).shift(shift)

            clean = VGroup(*[Dot(ax.coords_to_point(x, y), radius=0.07, color=C_ACCENT)
                              for x, y in zip(xs, ys)])
            out_d = Dot(ax.coords_to_point(2.5, 7.5), radius=0.11, color=C_RED)
            mu = ax.plot(lambda x, o=mu_off: x + o, x_range=[0, 5.1],
                         color=colour, stroke_width=2)
            up = ax.plot(lambda x, o=mu_off, b=band_w: x+o+2*b, x_range=[0, 5.1],
                         color=colour, stroke_width=0)
            lo = ax.plot(lambda x, o=mu_off, b=band_w: x+o-2*b, x_range=[0, 5.1],
                         color=colour, stroke_width=0)
            band = ax.get_area(up, bounded_graph=lo, color=colour, opacity=0.22)
            lbl = Text(label_str, font_size=17, color=colour, weight=BOLD)
            lbl.next_to(ax, UP, buff=0.15)
            return VGroup(ax, clean, out_d, mu, band, lbl)

        left  = make_panel(LEFT * 2.7,  C_RED,    1.4, 0.5,  "Gaussian")
        right = make_panel(RIGHT * 2.7, C_PURPLE, 0.45, 0.0, "Generalised")
        div   = DashedLine(UP * 3, DOWN * 3.5, color=C_GREY,
                           stroke_width=1, dash_length=0.18)

        for grp in [left, right]:
            self.play(Create(grp[0]))
            self.play(LaggedStart(*[GrowFromCenter(d) for d in grp[1]],
                                  lag_ratio=0.06, run_time=0.7))
            self.play(GrowFromCenter(grp[2]), Create(grp[3]),
                      FadeIn(grp[4]), Write(grp[5]))

        self.play(Create(div))

        ann_g = Text("Wide band · Mean shifted\nUnder-confident",
                     font_size=13, color=C_RED, line_spacing=1.2)
        ann_g.next_to(left[0], DOWN, buff=0.15)
        ann_p = Text("Tight band · Mean correct\nWell-calibrated",
                     font_size=13, color=C_PURPLE, line_spacing=1.2)
        ann_p.next_to(right[0], DOWN, buff=0.15)
        self.play(Write(ann_g), Write(ann_p))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════════════
# 12 – KEY TAKEAWAYS
# ════════════════════════════════════════════════════════════════════════════
class S12_Takeaways(Scene):
    def construct(self):
        title = Text("Key Takeaways", font_size=36, color=C_ACCENT, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        points = [
            ("1", "Single-pass only",
             "one forward pass → real-time on edge hardware", C_GREEN),
            ("2", "Gaussian NLL is fragile",
             "outliers → steep gradients → distorted μ and inflated σ", C_RED),
            ("3", "Generalised Normal unifies all",
             "β=2: Gaussian   β=1: Laplace   β→0: Cauchy (heavy tails)", C_PURPLE),
            ("4", "β is learnable automatically",
             "gradient descent adapts β to the outlier level — no manual tuning", C_YELLOW),
            ("5", "Tight, reliable uncertainty",
             "stable even with 50% outliers · validated on real robot grasping", C_ACCENT),
        ]

        grp = VGroup()
        for num, head, body, col in points:
            circle = Circle(radius=0.27, color=col, fill_color=col,
                            fill_opacity=0.2, stroke_width=1.5)
            n_txt  = Text(num, font_size=19, color=col).move_to(circle)
            h_txt  = Text(head, font_size=19, color=col, weight=BOLD)
            b_txt  = Text(body, font_size=14, color=C_GREY)
            h_txt.next_to(circle, RIGHT, buff=0.22)
            b_txt.next_to(h_txt, DOWN, buff=0.05, aligned_edge=LEFT)
            row = VGroup(circle, n_txt, h_txt, b_txt)
            grp.add(row)

        grp.arrange(DOWN, buff=0.30, aligned_edge=LEFT)
        grp.center().shift(DOWN * 0.1)

        for row in grp:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.55)

        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
