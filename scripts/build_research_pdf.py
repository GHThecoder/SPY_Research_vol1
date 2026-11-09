"""Generate a polished multi-page research PDF into reports/."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "research_report.pdf"
ALPHA_PATH = ROOT / "data" / "processed" / "alpha_candidates.yaml"

INK = colors.HexColor("#12141a")
MUTED = colors.HexColor("#5c6370")
RULE = colors.HexColor("#d8dce3")
ACCENT = colors.HexColor("#0f6e56")
PAPER = colors.HexColor("#f7f5f0")
CARD = colors.HexColor("#ffffff")


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=22,
            leading=28,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            textColor=INK,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def fig(path: Path, width: float = 15.5 * cm) -> list:
    if not path.exists():
        return [Paragraph(f"<i>[Missing figure: {path.name}]</i>", styles()["caption"])]
    img = Image(str(path))
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * aspect
    if img.drawHeight > 9.5 * cm:
        scale = (9.5 * cm) / img.drawHeight
        img.drawWidth *= scale
        img.drawHeight *= scale
    return [img, Paragraph(path.stem.replace("_", " "), styles()["caption"])]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(1.8 * cm, A4[1] - 1.2 * cm, A4[0] - 1.8 * cm, A4[1] - 1.2 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, A4[1] - 1.0 * cm, "SPY Alpha Research")
    canvas.drawRightString(A4[0] - 1.8 * cm, A4[1] - 1.0 * cm, "Proprietary")
    canvas.line(1.8 * cm, 1.3 * cm, A4[0] - 1.8 * cm, 1.3 * cm)
    canvas.drawCentredString(A4[0] / 2, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


def cover_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 0.45 * cm, A4[0], 0.45 * cm, fill=1, stroke=0)
    canvas.rect(0, 0, A4[0], 0.45 * cm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        A4[0] / 2,
        1.2 * cm,
        "Proprietary Source-Available · © 2026 · All rights reserved",
    )
    canvas.restoreState()


def kv_table(rows: list[tuple[str, str]], col_widths=None) -> Table:
    S = styles()
    data = [[Paragraph(f"<b>{k}</b>", S["bullet"]), Paragraph(v, S["bullet"])] for k, v in rows]
    t = Table(data, colWidths=col_widths or [4.5 * cm, 11.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CARD),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def load_alphas() -> list[str]:
    if not ALPHA_PATH.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(ALPHA_PATH.read_text(encoding="utf-8")) or {}
        return list(data.get("alphas") or [])
    except Exception:
        return []


def findings_table(S) -> Table:
    """Evidence-status table. Values are placeholders when figures/metrics not yet recomputed."""
    rows = [
        [
            Paragraph("<b>Claim</b>", S["bullet"]),
            Paragraph("<b>Effect</b>", S["bullet"]),
            Paragraph("<b>Evidence</b>", S["bullet"]),
        ],
        [
            Paragraph("Shortlist mean IC &gt; 0 (final 2024)", S["bullet"]),
            Paragraph("Point IC + block-bootstrap 95% CI", S["bullet"]),
            Paragraph("mixed — some moms supported; calendar/lags weak", S["bullet"]),
        ],
        [
            Paragraph("Equal-weight IC &gt; 0 (H2)", S["bullet"]),
            Paragraph("Bootstrap one-sided p on eqw", S["bullet"]),
            Paragraph("needs more data — not decisive on 2024 alone", S["bullet"]),
        ],
        [
            Paragraph("Equal-weight beats best single (H3)", S["bullet"]),
            Paragraph("Paired bootstrap ΔIC", S["bullet"]),
            Paragraph("not supported — singles can dominate eqw", S["bullet"]),
        ],
        [
            Paragraph("Eqw timing vs SPY B&amp;H / vol-matched", S["bullet"]),
            Paragraph("Final Sharpe / equity overlay", S["bullet"]),
            Paragraph("not supported — eqw may trail SPY benchmarks", S["bullet"]),
        ],
        [
            Paragraph("Feature-group necessity (ablation)", S["bullet"]),
            Paragraph("ΔIC / ΔSharpe vs full eqw", S["bullet"]),
            Paragraph("exploratory — see ablation_groups figure", S["bullet"]),
        ],
    ]
    t = Table(rows, colWidths=[5.5 * cm, 5.5 * cm, 5.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2ee")),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build():
    S = styles()
    story: list = []

    story.append(Spacer(1, 3.2 * cm))
    story.append(Paragraph("SPY ALPHA RESEARCH", S["meta"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Interpretable Feature Attribution<br/>and Alpha Validation<br/>for SPY Return Dynamics",
            S["cover_title"],
        )
    )
    story.append(HRFlowable(width="60%", thickness=1.2, color=ACCENT, spaceBefore=6, spaceAfter=12))
    story.append(
        Paragraph(
            "Three-way design: train → selection shortlist → one-shot final 2024 validation.<br/>"
            "Primary IC inference uses moving-block bootstrap; rolling IC is for stability plots only.",
            S["cover_sub"],
        )
    )
    story.append(Spacer(1, 1.0 * cm))
    story.append(
        kv_table(
            [
                ("Instrument", "SPY (US equity ETF)"),
                ("Sample", "2016-01-01 → 2024-12-31 (hard cap; no 2025+)"),
                ("Train", "2016–2021 — fit + purged inner CV"),
                ("Selection", "2022–2023 — attribution, Borda, model compare"),
                ("Final", "2024 — one-shot IC / hypotheses / backtests"),
                ("Scaling", "Causal expanding time-series z-score"),
                ("License", "Proprietary Source-Available"),
                ("Version", "0.3.0 (research-grade hardening)"),
            ]
        )
    )
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Research Report", S["meta"]))
    story.append(PageBreak())

    story.append(Paragraph("1. Executive summary", S["h1"]))
    story.append(
        Paragraph(
            "This report documents a two-stage research program on SPY return dynamics with an "
            "explicit <b>three-period</b> calendar. Stage A fits models on 2016–2021 and evaluates "
            "attribution on the <b>selection</b> window (2022–2023). A Borda shortlist is frozen "
            "there. Stage B evaluates information coefficients, pre-registered hypotheses, equal-weight "
            "timing, benchmarks, and grouped ablation on <b>final 2024 only</b> — never used for "
            "selection. Primary IC inference is moving-block bootstrap on the level series; Newey–West "
            "on demeaned daily contributions is secondary. Rolling IC is retained for stability plots "
            "and must not be used for significance stars.",
            S["body"],
        )
    )
    story.append(Paragraph("Core principle", S["h2"]))
    story.append(
        Paragraph(
            "Forecast accuracy is secondary. The north star is <b>stability of explanation</b> and "
            "whether selected features survive as tradeable timing signals after statistical and "
            "cost-aware validation — including honest negative findings versus SPY benchmarks.",
            S["body"],
        )
    )

    story.append(Paragraph("2. Data, contract, and three-way split", S["h1"]))
    story.append(
        Paragraph(
            "Features at date <font face='Courier'>t</font> use information known at the "
            "<b>close of day t</b>. The regression / IC target is always the forward return "
            "<font face='Courier'>r<sub>t+1</sub></font>. ExpandingScaler (and optional RollingScaler) "
            "carry training history into OOS transforms. Combined 2022–2024 analyses, if present, "
            "are labeled exploratory and are not the basis for the shortlist.",
            S["body"],
        )
    )
    story.append(
        kv_table(
            [
                ("Universe", "SPY OHLCV via yfinance → parquet cache"),
                ("Target", "1-day forward log return"),
                ("Feature families", "return lags, momentum, realized vol, vol-of-vol, range, volume z, calendar"),
                ("Inner CV", "Purged expanding walk-forward on train"),
                ("Alpha shortlist", "Borda consensus, top_k = 8, selection attribution only"),
            ]
        )
    )

    story.append(Paragraph("3. Classical structure", S["h1"]))
    story.append(
        Paragraph(
            "Before machine-learning attribution, serial dependence in SPY returns is documented "
            "with ACF/PACF and an ARIMA order grid on the training window.",
            S["body"],
        )
    )
    story.extend(fig(FIG / "acf_pacf.png"))

    story.append(PageBreak())
    story.append(Paragraph("4. Attribution pipelines (selection window)", S["h1"]))
    story.append(
        Paragraph(
            "Primary attribution surfaces are computed on <b>2022–2023 selection</b> after fitting "
            "on 2016–2021. Final 2024 is untouched until Stage B.",
            S["body"],
        )
    )
    story.append(Paragraph("4.1 Tree models (RF / HistGBM)", S["h2"]))
    story.extend(fig(FIG / "tree_importance.png"))
    story.append(Paragraph("4.2 PCA + linear / kernel", S["h2"]))
    story.extend(fig(FIG / "pca_loadings.png"))
    story.extend(fig(FIG / "kernel_perm_importance.png"))
    story.append(Paragraph("4.3 CNN feature extractor", S["h2"]))
    story.extend(fig(FIG / "cnn_saliency.png"))
    story.append(Paragraph("4.4 Rank agreement", S["h2"]))
    story.extend(fig(FIG / "rank_stability.png"))
    story.append(Paragraph("4.5 Model comparison (selection)", S["h2"]))
    story.extend(fig(FIG / "model_comparison.png"))

    story.append(PageBreak())
    story.append(Paragraph("5. Consensus alpha shortlist", S["h1"]))
    story.append(
        Paragraph(
            "Selection-window importances from Random Forest and PCA-Ridge are fused with Borda "
            "consensus. The shortlist is persisted before any final-window peek and is the only "
            "input to notebooks 08–09.",
            S["body"],
        )
    )
    alphas = load_alphas() or [
        "dow",
        "hl_range",
        "mom_10",
        "mom_126",
        "mom_21",
        "mom_63",
        "month",
        "ret_lag_1",
    ]
    alpha_rows = [[Paragraph("<b>#</b>", S["bullet"]), Paragraph("<b>Selected alpha</b>", S["bullet"])]]
    for i, a in enumerate(alphas, 1):
        alpha_rows.append(
            [Paragraph(str(i), S["bullet"]), Paragraph(f"<font face='Courier'>{a}</font>", S["bullet"])]
        )
    at = Table(alpha_rows, colWidths=[1.5 * cm, 14.5 * cm])
    at.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2ee")),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(at)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("6. IC, inference, and hypotheses (final 2024)", S["h1"]))
    story.append(
        Paragraph(
            "Each selected feature is mapped to a causal expanding z-score with direction fixed from "
            "the <b>train</b> IC sign. Primary inference: full-window Pearson IC with moving-block "
            "bootstrap CI and one-sided p for IC &gt; 0. Secondary: Newey–West on the demeaned "
            "product series. BH-FDR is applied to bootstrap p-values. Pre-registered tests: "
            "H1 mean IC(alpha) &gt; 0; H2 mean IC(equal-weight) &gt; 0; "
            "H3 IC(equal-weight) ≥ IC(best single) via paired block bootstrap.",
            S["body"],
        )
    )
    story.extend(fig(FIG / "ic_by_alpha.png"))
    story.extend(fig(FIG / "rolling_ic_stability.png"))

    story.append(PageBreak())
    story.append(Paragraph("7. Final backtest, benchmarks, ablation", S["h1"]))
    story.append(
        Paragraph(
            "Equal-weight and single-alpha timing use identical "
            "<font face='Courier'>signal_lag</font> and costs. Benchmarks on final 2024: SPY "
            "buy-and-hold, cash, and volatility-matched SPY with scale fit on <b>selection only</b>. "
            "Grouped ablation drops momentum / lag / vol / volume / calendar blocks from the shortlist.",
            S["body"],
        )
    )
    story.extend(fig(FIG / "eqw_vs_singles_equity.png"))
    story.extend(fig(FIG / "benchmarks_equity.png"))
    story.extend(fig(FIG / "all_signals_compare.png"))
    story.extend(fig(FIG / "ablation_groups.png"))

    story.append(Paragraph("8. Findings status and caveats", S["h1"]))
    story.append(
        Paragraph(
            "Shortlist membership was chosen on <b>selection</b>, not cherry-picked on final Sharpe. "
            "Negative findings are first-class: equal-weight need not beat singles or SPY benchmarks. "
            "Evidence labels below are the interpretive contract for this sample — recompute after "
            "re-running notebooks 08–09 before citing numbers externally.",
            S["body"],
        )
    )
    story.append(findings_table(S))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Caveats", S["h2"]))
    caveats = [
        "Prior joint 2022–2024 analyses (if any residual) are exploratory only — selection bias history.",
        "Final 2024 is a single calendar year; power is limited — many claims correctly stay 'needs more data'.",
        "SPY-only univariate timing; no cross-sectional L/S book or multi-asset confirmation.",
        "Costs, lag, and position map are assumptions; results are sensitive to those knobs.",
        "This hardening does not claim elite-desk production readiness.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, S["bullet"]), leftIndent=8, bulletColor=ACCENT) for b in caveats],
            bulletType="bullet",
            start="•",
        )
    )

    story.append(Paragraph("Reproducibility checklist", S["h2"]))
    bullets = [
        "Three-way dates in configs/experiment.yaml (final never used for shortlist)",
        "Causal scaler + feature timestamp assert in notebook 02",
        "Selection attribution in notebooks 04–07; final-only 08–09",
        "Block-bootstrap IC primary; rolling IC plots only",
        "Benchmarks: SPY B&H, cash, vol-matched (scale on selection)",
        "pytest + requirements.lock + restart Jupyter kernel after pulls",
        "Proprietary Source-Available LICENSE in repository root",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, S["bullet"]), leftIndent=8, bulletColor=ACCENT) for b in bullets],
            bulletType="bullet",
            start="•",
        )
    )

    story.append(Spacer(1, 1.0 * cm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=8))
    story.append(
        Paragraph(
            "© 2026 · Proprietary Source-Available License · All rights reserved.<br/>"
            "Generated by <font face='Courier'>scripts/build_research_pdf.py</font>",
            S["caption"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="SPY Alpha Research — Interpretable Feature Attribution Report",
        author="SPY Alpha Research",
    )
    doc.build(story, onFirstPage=cover_footer, onLaterPages=header_footer)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
