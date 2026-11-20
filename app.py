"""Streamlit interface for the local Treasury Risk Lab."""

import altair as alt
import pandas as pd
import streamlit as st

from bond_pricing import DEFAULT_PORTFOLIO, build_portfolio, portfolio_summary
from data_pipeline import (
    CORE_TENORS, DATA_PATH, PCA_TENORS, PORTFOLIO_TENORS, compute_spreads,
    filter_date_range, get_yield_curve_for_date, load_raw_csv,
)
from historical_var import compute_historical_scenarios, compute_var_es, risk_contribution_by_tenor
from monte_carlo import run_monte_carlo
from pca_analysis import interpret_components, run_pca
from stress_testing import PRESET_SCENARIOS, run_stress_scenario


@st.cache_data
def get_data():
    return load_raw_csv()


def money(value):
    return f"${value:,.0f}" if value >= 0 else f"−${abs(value):,.0f}"


def risk_table(result):
    return pd.DataFrame([
        {
            "Confidence": f"{level}%",
            "VaR (USD)": result[f"var_{level}_dollar"],
            "ES (USD)": result[f"es_{level}_dollar"],
            "VaR (% of value)": result[f"var_{level}_pct"],
            "ES (% of value)": result[f"es_{level}_pct"],
        }
        for level in (95, 99)
    ])


def market_view(curve, yields, history):
    metrics = st.columns(3)
    metrics[0].metric("2Y par yield", f"{yields['2 Yr']:.2f}%")
    metrics[1].metric("10Y par yield", f"{yields['10 Yr']:.2f}%")
    spread = (yields["10 Yr"] - yields["2 Yr"]) * 100
    metrics[2].metric("10Y − 2Y spread", f"{spread:+.0f} bp")

    st.subheader("Treasury par yield curve")
    chart = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("tenor_years:Q", title="Maturity (years)"),
        y=alt.Y("yield_pct:Q", title="Par yield (%)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("tenor_label:N", title="Tenor"),
                 alt.Tooltip("yield_pct:Q", title="Par yield (%)", format=".2f")],
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Yield and spread history")
    st.caption("Monthly average par yields and daily 10Y − 2Y spread")
    # Keep the historical overview readable and below the default chart row limit.
    monthly = history.set_index("date")[CORE_TENORS].resample("ME").mean().reset_index()
    # An incomplete last month is reported with its available observations only.
    monthly["date"] = monthly["date"].clip(upper=history["date"].max())
    trend = monthly.melt("date", CORE_TENORS, var_name="Tenor", value_name="Yield")
    chart = alt.Chart(trend).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Yield:Q", title="Par yield (%)"),
        color=alt.Color("Tenor:N", sort=CORE_TENORS, legend=alt.Legend(orient="top")),
    )
    st.altair_chart(chart, use_container_width=True)
    spreads = compute_spreads(history)
    chart = alt.Chart(spreads).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("10Y_2Y:Q", title="10Y − 2Y (bp)"),
    )
    st.altair_chart(chart, use_container_width=True)


def portfolio_view(yields, spec):
    portfolio = build_portfolio(yields, spec)
    summary = portfolio_summary(portfolio)
    metrics = st.columns(4)
    metrics[0].metric("Market value", money(summary["total_market_value"]))
    metrics[1].metric("Modified duration", f"{summary['portfolio_modified_duration']:.2f} yr")
    metrics[2].metric("DV01", f"{money(summary['portfolio_dv01'])}/bp")
    metrics[3].metric("Convexity", f"{summary['portfolio_convexity']:.2f} yr²")

    columns = {
        "tenor": "Tenor", "face_value": "Face (USD)", "weight": "Weight",
        "coupon_rate": "Coupon", "market_value": "Value (USD)",
        "modified_duration": "Mod. duration (yr)", "dv01": "DV01 (USD/bp)",
    }
    detail = portfolio[list(columns)].rename(columns=columns)
    st.dataframe(
        detail.style.format({
            "Face (USD)": "{:,.0f}", "Weight": "{:.1%}", "Coupon": "{:.2%}",
            "Value (USD)": "{:,.0f}", "Mod. duration (yr)": "{:.2f}",
            "DV01 (USD/bp)": "{:,.2f}",
        }), hide_index=True, use_container_width=True,
    )
    st.caption("Coupons and initial YTMs equal the as-of par yields. Bonds pay semiannually on a regular coupon schedule.")


def stress_view(yields, spec):
    st.subheader("Curve shock")
    preset = st.selectbox("Scenario", [*PRESET_SCENARIOS, "Custom shock"])
    if preset == "Custom shock":
        shocks = {tenor: col.slider(f"{tenor} (bp)", -300, 300, 0, 5)
                  for col, tenor in zip(st.columns(3), PORTFOLIO_TENORS)}
    else:
        shocks = PRESET_SCENARIOS[preset]

    result = run_stress_scenario(yields, shocks, spec)
    metrics = st.columns(3)
    metrics[0].metric("Full repricing P&L", money(result["total_pnl_dollar"]),
                      f"{result['total_pnl_pct']:+.2f}%", delta_color="off")
    metrics[1].metric("Duration + convexity P&L", money(result["duration_approx_pnl"]))
    metrics[2].metric("Full less approximated P&L", money(result["approximation_error"]))

    columns = {
        "tenor": "Tenor", "shock_bp": "Shock (bp)", "old_price": "Base value (USD)",
        "new_price": "Shocked value (USD)", "pnl_dollar": "Full P&L (USD)",
        "duration_approx_pnl": "Approx. P&L (USD)",
    }
    detail = result["position_details"][list(columns)].rename(columns=columns)
    st.dataframe(
        detail.style.format({col: "{:,.0f}" for col in detail.columns[1:]}),
        hide_index=True, use_container_width=True,
    )
    st.caption("Only yields are shocked. Coupon rates, remaining maturities, and face values stay fixed.")


@st.cache_data
def get_historical(history, as_of, start, yields, spec):
    scenarios = compute_historical_scenarios(history, as_of, yields, spec, start_date=start)
    attribution = risk_contribution_by_tenor(history, as_of, yields, spec, start_date=start)
    return scenarios, compute_var_es(scenarios), attribution


def historical_view(history, as_of, start, yields, spec):
    st.subheader("Historical VaR and Expected Shortfall")
    scenarios, result, attribution = get_historical(history, as_of, start, yields, spec)
    st.caption(
        f"{len(scenarios):,} aligned daily shocks · "
        f"{scenarios['scenario_date'].min():%Y-%m-%d} to {scenarios['scenario_date'].max():%Y-%m-%d} · "
        f"{len(scenarios) * 0.01:.1f} observation-equivalents in the 99% tail"
    )
    show_risk_table(result)
    st.caption("VaR is the empirical loss quantile. ES averages the worst tail at the stated confidence. Positive values denote losses.")

    chart = alt.Chart(scenarios).mark_bar().encode(
        x=alt.X("pnl_dollar:Q", bin=alt.Bin(maxbins=45), title="Scenario P&L (USD)"),
        y=alt.Y("count():Q", title="Scenarios"),
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("95% Expected Shortfall by tenor")
    detail = attribution[["tenor", "es_contribution_95"]].rename(
        columns={"tenor": "Tenor", "es_contribution_95": "ES contribution (USD)"}
    )
    st.dataframe(detail.style.format({"ES contribution (USD)": "{:,.2f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Each tenor uses the same portfolio loss-tail scenarios and weights. Its signed contribution can offset losses on other tenors.")
    st.download_button("Download historical scenarios", scenarios.to_csv(index=False),
                       file_name=f"historical_scenarios_{as_of}.csv", mime="text/csv")


def show_risk_table(result):
    formats = {"VaR (USD)": "{:,.0f}", "ES (USD)": "{:,.0f}",
               "VaR (% of value)": "{:.3f}", "ES (% of value)": "{:.3f}"}
    st.dataframe(risk_table(result).style.format(formats),
                 hide_index=True, use_container_width=True)


@st.cache_data
def get_pca(history, start, as_of):
    return run_pca(history, start_date=start, end_date=as_of)


def pca_view(history, start, as_of):
    result = get_pca(history, start, as_of)
    st.subheader("Principal components")
    components = [f"PC{i + 1}" for i in range(len(result["explained_variance_ratio"]))]
    for col, name, ratio in zip(st.columns(len(components)), components, result["explained_variance_ratio"]):
        col.metric(f"{name} variance explained", f"{ratio:.1%}")

    loadings = result["loadings_df"].rename_axis("Tenor").reset_index()
    loadings = loadings.melt(["Tenor", "tenor_years"], components, "Component", "Loading")
    chart = alt.Chart(loadings).mark_line(point=True).encode(
        x=alt.X("tenor_years:Q", title="Maturity (years)"),
        y=alt.Y("Loading:Q", title="Eigenvector loading"),
        color=alt.Color("Component:N", legend=alt.Legend(orient="top")),
        tooltip=["Component:N", "Tenor:N", alt.Tooltip("Loading:Q", format=".3f")],
    )
    st.altair_chart(chart, use_container_width=True)
    dates = result["changes_df"]["date"]
    st.caption(
        f"Covariance PCA · {result['n_observations']:,} aligned daily shocks · "
        f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d} · Tenors {', '.join(PCA_TENORS)}"
    )
    for note in interpret_components(result["loadings_df"]):
        st.caption(note)


@st.cache_data
def get_monte_carlo(history, yields, spec, start, as_of, num_simulations):
    return run_monte_carlo(
        history, yields, spec, n_simulations=num_simulations,
        start_date=start, valuation_date=as_of,
    )


def monte_carlo_view(history, yields, spec, start, as_of):
    st.subheader("Gaussian Monte Carlo")
    count = st.select_slider("Simulated daily shocks", [1_000, 5_000, 10_000], value=5_000)
    st.caption(f"Calibration window {start} to {as_of} · Random seed 42")
    result = get_monte_carlo(history, yields, spec, start, as_of, count)
    show_risk_table(result)
    st.caption("Shocks follow a multivariate normal distribution fitted to observed yield changes. Extreme rate moves may be underrepresented.")


def main():
    st.set_page_config(page_title="Treasury Risk Lab", layout="wide", initial_sidebar_state="collapsed")
    st.title("Treasury Risk Lab")
    st.caption("U.S. Treasury par yields and rate risk for a hypothetical 2Y / 5Y / 10Y bond portfolio")
    if not DATA_PATH.is_file():
        st.error(f"Missing historical dataset: {DATA_PATH.name}. See the README for the source.")
        return

    data = get_data()
    valid = data.dropna(subset=PORTFOLIO_TENORS)
    if valid.empty:
        st.error("No observation has complete 2Y, 5Y, and 10Y yields.")
        return
    view = st.radio(
        "Analysis", ["Market", "Portfolio", "Stress test", "Historical risk", "PCA", "Monte Carlo"],
        horizontal=True, label_visibility="collapsed",
    )

    st.sidebar.header("Valuation")
    requested = st.sidebar.date_input(
        "Valuation date", valid["date"].max().date(),
        min_value=valid["date"].min().date(), max_value=valid["date"].max().date(),
        help="Uses the latest complete curve on or before the requested date.",
    )
    curve = get_yield_curve_for_date(valid, requested)
    observed = pd.Timestamp(curve["date"].iloc[0])
    as_of = observed.date().isoformat()
    start = max(data["date"].min(), observed - pd.DateOffset(years=7) + pd.Timedelta(days=1))
    start = start.date().isoformat()
    yields = dict(zip(curve["tenor_label"], curve["yield_pct"]))
    st.markdown(f"**Curve as of {as_of}**")
    st.caption("Open the sidebar to change the date or portfolio face amounts.")
    if observed.date() != requested:
        st.info(f"Used {as_of}, the latest complete observation on or before {requested}.")
    history = filter_date_range(data, start, as_of)

    st.sidebar.header("Portfolio face amounts")
    spec = []
    for position in DEFAULT_PORTFOLIO:
        face = st.sidebar.number_input(
            f"{position['tenor']} face (USD)", 0, 50_000_000,
            int(position["face_value"]), step=100_000,
        )
        spec.append({**position, "face_value": face})

    try:
        if view == "Market":
            market_view(curve, yields, history)
        elif view == "PCA":
            pca_view(history, start, as_of)
        elif not any(position["face_value"] for position in spec):
            st.info("Enter a positive face amount to run portfolio analysis.")
        elif view == "Portfolio":
            portfolio_view(yields, spec)
        elif view == "Stress test":
            stress_view(yields, spec)
        elif view == "Historical risk":
            historical_view(history, as_of, start, yields, spec)
        else:
            monte_carlo_view(history, yields, spec, start, as_of)
    except ValueError as exc:
        st.warning(str(exc))


if __name__ == "__main__":
    main()
