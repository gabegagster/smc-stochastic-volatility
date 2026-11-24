# Presentation Speaker Notes — Cholesky Multivariate Stochastic Volatility (ChMSV)

**Course:** Sequential Monte Carlo (`particles`) — Prof. Nicolas Chopin (ENSAE Paris)  
**Presentation Date:** 17 December 2026  
**Estimated Total Time (Implemented Slides):** ~11 minutes (10 min 50 s across 16 slides) + Q&A  
**Reference Paper:** Szczepocki, P. (2023). *Estimation of the Cholesky Multivariate Stochastic Volatility Model Using Iterated Filtering.* Econometrics. Ekonometria. Advances in Applied Data Analysis, 27(4), 44–58.  
**Deck Files:** `presentation/smc_chmsv_ensae_presentation.pptx` / `presentation/smc_chmsv_ensae_presentation.pdf`

> **Note:** Speaker roles are left unassigned. Notes below cover the implemented slides (**Slides 1–13** and **Slides 18–20**); upcoming estimation slides (**Slides 14–17**) are omitted until those modules are implemented.

---

## Slide-by-Slide Speaker Notes

### Slide 1 — Sequential Monte Carlo: Project Presentation *(~25s)*
* **Spoken Script:**  
  "Good afternoon Professor Chopin and everyone. Today we are presenting our project on the **Cholesky Multivariate Stochastic Volatility model**, or ChMSV, implemented in Python using the `particles` package and based on Szczepocki (2023)."
* **Visual Cue:** Gesture briefly to the title and team line.
* **Transition:** "Let's start with a quick overview of the project structure."

---

### Slide 2 — Agenda *(~20s)*
* **Spoken Script:**  
  "Here is our roadmap: first, the financial motivation and empirical data setup; second, formulating the ChMSV model as a state-space model in `particles` alongside synthetic benchmarks; third, live state tracking with a Bootstrap Particle Filter and our empirical safe-haven findings; and finally, evaluation and conclusions."
* **Transition:** "Let's begin with why dynamic correlation is needed in practice."

---

### Slide 3 — Use Case & Motivation *(~45s)*
* **Spoken Script:**  
  "In portfolio risk management, computing one static Pearson correlation over a 5-year window hides what actually happens during market crashes. Two assets can look uncorrelated on average while suddenly moving together in a panic—precisely when diversification is needed most.  
  Following Baur and Lucey (2009), a **strong safe haven** must be *negatively* correlated with equities during market turmoil, whereas a **weak safe haven** is simply uncorrelated during stress. To test whether Gold, Bitcoin, or Ethereum protect an S&P 500 portfolio, we need **time-varying** volatility and correlation—which are unobserved latent states on any given day.  
  The main technical bottleneck in multi-asset models is guaranteeing that the conditional covariance matrix $\Sigma_t$ stays **positive definite** every single day. The ChMSV model solves this cleanly using an $LDL^\top$ Cholesky decomposition, allowing us to model all four assets jointly in `particles`."
* **Visual Cue:** Point to the right-hand diagram: 3 asset boxes flowing down into the unified 4D ChMSV block ($T = 1{,}258$, $10$ latent states, $30$ parameters).
* **Transition:** "Here is how the codebase is organized into independent modules."

---

### Slide 4 — Project Scope & Architecture *(~30s)*
* **Spoken Script:**  
  "Our repository in `src/` is split into four modular phases. The first two—data replication with the vectorized `ChMSV` state-space model, and live state tracking with the Bootstrap Particle Filter—are fully implemented and verified, while the two parameter-estimation phases will plug directly into the same `ChMSV` interface."
* **Visual Cue:** Point to the two active cards on the left (`Phase I` and `Phase II`).
* **Transition:** "Let's zoom in on Phase I, starting with the empirical dataset."

---

### Slide 5 — Input Datasets & Profiling *(~55s)*
* **Spoken Script:**  
  "We pulled daily prices from Stooq.pl from **4 June 2018 to 4 June 2023**, keeping only common trading days across all four series—giving $T = 1{,}258$ daily percentage log-returns.  
  In the top-left table and top-right bar chart, notice two contrasting risk profiles: Gold is the calmest asset with a daily standard deviation of `0.926%`, while Bitcoin (`4.302%`) and Ethereum (`5.693%`) have 4 to 6 times higher baseline volatility. Yet **SPX** has by far the highest kurtosis (`16.221`)—calm most days, with extreme crash outliers in March 2020.  
  Comparing against Szczepocki's Table 1 revealed a subtle data-engineering catch: SPX matches to the exact third decimal place, and spot Gold (`xauusd`) matches nicely, whereas standard `00:00 UTC` crypto feeds distort Ethereum's 12 March 2020 crash to `-55.07%`. Aligning to Stooq's **Warsaw midnight (`00:00 CET`)** close reproduces the paper's `-42.86%` drop (`-42.45%` here) and kurtosis (`8.55` vs. `8.59`)."
* **Visual Cue:** Point to the SPX bar (`16.22` kurtosis) in the chart, then the bottom-right alignment card.
* **Transition:** "Let's look at the actual price and return time series across those five years."

---

### Slide 6 — Empirical Data: Prices & Log-Returns (Fig. 1) *(~35s)*
* **Spoken Script:**  
  "This plot reproduces Figure 1 of the paper: asset prices on the left and percentage log-returns on the right. Visually, you can see volatility clustering immediately—SPX and Gold show a massive, concentrated spike right at the March 2020 COVID shock, whereas Bitcoin and Ethereum exhibit sustained, wide swings across 2019–2023.  
  We exported both a full 4-asset dataset and a fast 2-asset subset (`SPX` and `Gold`) into `data/processed/`."
* **Visual Cue:** Point to the March 2020 spike in SPX/Gold returns (right column) vs. the persistent amplitude in BTC/ETH.
* **Transition:** "Now let's look at how the ChMSV model turns those four return series into a clean state-space model."

---

### Slide 7 — ChMSV State-Space Specification *(~60s)*
* **Spoken Script:**  
  "Think of ChMSV as a continuous-state Hidden Markov Model. At each day $t$, Equation 1 generates observed returns $\mathbf{y}_t$ by scaling standard normal noise $\mathbf{e}_t$ through $\mathbf{L}_t \mathbf{V}_t^{1/2}$, giving a conditional Gaussian with covariance $\Sigma_t = \mathbf{L}_t \mathbf{V}_t \mathbf{L}_t^\top$.  
  The key linear-algebra trick is that $\mathbf{V}_t$ has $\exp(h_{i,t})$ on its diagonal—guaranteed strictly positive—and $\mathbf{L}_t$ is unit lower-triangular holding unconstrained elements $q_{ij,t}$. Because any real numbers work, we convert a constrained $4 \times 4$ matrix problem into $d_x = p(p+1)/2 = 10$ unconstrained, independent Gaussian AR(1) hidden states (Equation 2). Once filtered, Equations 6 and 7 reconstruct $\Sigma_t$ to read off daily standard deviations and dynamic correlations.  
  On the right, this maps 1-to-1 into `particles` inside `src/base_model.py`: `ChMSVInitDist` starts at the unconditional mean $\boldsymbol{\mu}$ (a Dirac mass as in the paper), `ChMSVTransDist` samples the 10 diagonal Gaussians in one vectorized call, and `ChMSVObsDist` evaluates the emission likelihood via a fast batched triangular Cholesky solve."
* **Visual Cue:** Walk down Equations 1, 2, and 3 on the left, then down boxes 1–4 on the right.
* **Transition:** "Before touching real market data, we built synthetic benchmarks to verify our filters against known ground truth."

---

### Slide 8 — Simulated Ground-Truth Benchmarks *(~40s)*
* **Spoken Script:**  
  "Using the exact parameters reported in Szczepocki's Table 2 as ground truth, we generated two synthetic datasets: a fast **2-asset** dataset (`p = 2`, 3 states, 9 parameters, $T = 500$) for rapid prototyping, and a full **4-asset** dataset (`p = 4`, 10 states, 30 parameters, $T = 1{,}258$).  
  Because we store the exact true hidden volatilities and true dynamic correlations alongside the noisy returns, we can directly measure filtering RMSE and estimation error before running on real data."
* **Visual Cue:** Point to the configuration table on the left and the 4-panel ground-truth trajectories on the right.
* **Transition:** "Next, we test live state tracking on this simulated benchmark using the Bootstrap Particle Filter."

---

### Slide 9 — Bootstrap Particle Filter & Simulated Tracking *(~55s)*
* **Spoken Script:**  
  "In `src/filtering.py`, we implemented the Bootstrap Particle Filter using `particles.SMC` with `ssm.Bootstrap`, $N = 1{,}000$ particles, systematic resampling, and an adaptive threshold $\mathrm{ESS}_t < 0.5\,N$. From the weighted particle cloud at each day $t$, we extract both the filtered mean and the 5th–95th percentile credibility bands.  
  On the simulated 4-asset benchmark, the filter tracks the true latent states closely: volatility RMSE is `0.180` for SPX and `0.379` for Gold, and dynamic correlation RMSE for SPX–Gold is just `0.026`. On the bottom-right panel, whenever a large return shock hits, only a few particles fit the observation, dropping ESS below `500` and immediately triggering resampling to prevent weight collapse—giving a healthy average ESS of `401.3 / 1000`."
* **Visual Cue:** Point to the dashed black true curves inside the shaded 90% bands, the RMSE table on the bottom left, and the red `500` ESS threshold line.
* **Transition:** "Next, we verified the theoretical Monte Carlo convergence rate and benchmarked our log-likelihood against Szczepocki's Table 2."

---

### Slide 10 — Monte Carlo Sensitivity & Table 2 Verification *(~60s)*
* **Spoken Script:**  
  "First, on the left, we ran 10 independent filter passes across $N \in \{250, 500, 1000, 2000\}$ on the simulated 4D data. As predicted by the SMC Central Limit Theorem, the standard deviation of $\log \hat{L}$ decays cleanly at rate $\mathcal{O}(N^{-1/2})$—dropping from `7.56` at $N=250$ down to `3.66` at $N=1000$ and `2.89` at $N=2000$—closely tracking the orange theoretical curve while mean ESS stays around `40%` of $N$.  
  Second, on the top right, we ran 10 independent filters at $N=1{,}000$ on the **real** 4-asset dataset at Table 2 parameters. Notice an interesting finding: while adaptive resampling ($\gamma = 0.5$) works smoothly most days, skipping resampling right before an extreme March 2020 crash amplifies weight collapse (`Std = 9.11`). Resampling every step ($\gamma = 1.0$, matching R's `pomp::pfilter` used in the paper) cuts single-run standard deviation in half to `4.16` and gives a `logmeanexp` of `-9975.17` with jackknife SE `1.12`—closely reproducing the paper's `-9972.73` (`SE = 0.65` over 12 runs)."
* **Visual Cue:** Point to the dashed amber CLT curve on the left, then compare the `Adap. (γ=0.5)` and `Std. (γ=1.0)` columns in the right table.
* **Transition:** "Now let's examine the filtered volatilities and correlations extracted from the real 2018–2023 market data."

---

### Slide 11 — Empirical Results: Return Volatilities (Fig. 2) *(~45s)*
* **Spoken Script:**  
  "Here is our reproduction of Figure 2 from the paper—daily conditional standard deviations along with 90% particle credibility bands.  
  The filter cleanly captures the hierarchy of market risk: **SPX** stays low around `0.7%–1.5%` on calm days before spiking abruptly near `3.0%` during the March 2020 COVID shock; **Gold** is the most stable series (`0.7%–1.1%`, peaking milder near `2.6%`); while **Bitcoin** (`2%–12%`) and **Ethereum** (`3%–16%`) exhibit structurally elevated volatility with frequent multi-month bursts throughout the five-year sample."
* **Visual Cue:** Point to the sharp March 2020 peak in SPX (top panel) vs. the persistent spikes in BTC and ETH (bottom two panels).
* **Transition:** "Turning to the core economic question of the paper: did any of these three candidates act as a safe haven for the S&P 500?"

---

### Slide 12 — Empirical Results: SPX Correlations (Fig. 3) *(~50s)*
* **Spoken Script:**  
  "This reproduces Figure 3—dynamic pairwise correlations between S&P 500 and each candidate, with `0` marked by the dashed horizontal line.  
  Notice two clear results: first, **all three correlations stay strictly positive** over the entire 2018–2023 period—roughly `0.03–0.13` for Gold, `0.02–0.21` for Bitcoin, and `0.01–0.14` for Ethereum. Second, all three correlations actually **reach their peak during the March 2020 equity crash**.  
  By Baur and McDermott's definition, none of Gold, Bitcoin, or Ethereum behaves as a **strong safe haven**, because none turns negative in a crisis. Because correlations stay modest overall, Gold remains the best diversifier thanks to its low volatility and low crisis co-movement."
* **Visual Cue:** Point to the dashed zero baseline at the bottom of each panel and the March 2020 peak across all three subplots.
* **Transition:** "Unlike prior bivariate studies, modeling all four assets jointly also reveals how the safe-haven candidates interact with one another."

---

### Slide 13 — Empirical Results: Cross-Correlations (Fig. 4) *(~45s)*
* **Spoken Script:**  
  "Figure 4 plots the three cross-correlations among the candidates themselves. Gold shows low-to-moderate correlation with both Bitcoin and Ethereum, briefly dipping **negative** during episodes in late 2018, mid-2019, and early 2022—offering genuine cross-asset diversification.  
  In sharp contrast, **Bitcoin and Ethereum** remain strongly and noisily co-moving—roughly `0.40 to 0.95`—meaning holding both cryptocurrencies adds very little diversification benefit relative to one another."
* **Visual Cue:** Contrast the green/cyan Gold–Crypto lines dipping below `0` against the pink BTC–ETH curve hovering near `0.4–0.95`.
* **Transition:** "Let's wrap up with the main methodological insights and limitations observed so far."

---

*(Slides 14–17 are upcoming parameter-estimation placeholders and are not included in these notes yet.)*

---

### Slide 18 — Evaluation: Discussion & Limitations *(~40s)*
* **Spoken Script:**  
  "Three key methodological takeaways emerge from our implementation so far:  
  First, **parameter interpretability**: raw AR(1) parameters of $q_{ij,t}$ mix variances and covariances, so economic insight always requires reconstructing $\Sigma_t = \mathbf{L}_t \mathbf{V}_t \mathbf{L}_t^\top$.  
  Second, **10D weight degeneracy**: in high-dimensional financial series with extreme shocks like March 2020, resampling every step ($\gamma = 1.0$) proved critical—cutting Monte Carlo single-run variance in half compared to $\gamma = 0.5$.  
  Third, **specification boundaries**: the baseline ChMSV assumes symmetric Gaussian errors—meaning natural future extensions include the **leverage effect** (negative returns triggering larger volatility spikes) and **heavy-tailed Student-$t$ innovations**."
* **Visual Cue:** Walk through items 1–3 top to bottom.
* **Transition:** "To conclude..."

---

### Slide 19 — Conclusion *(~35s)*
* **Spoken Script:**  
  "To summarize what we have completed:  
  - We aligned 1,258 daily Stooq returns across SPX, Gold, Bitcoin, and Ethereum matching Table 1 and implemented vectorized 2D and 4D `ChMSV` models in `particles` alongside synthetic benchmarks.  
  - We validated our Bootstrap Particle Filter on synthetic truth (`0.180` SPX vol RMSE, `0.026` corr RMSE), confirmed $\mathcal{O}(N^{-1/2})$ CLT scaling, reproduced Table 2's log-likelihood (`-9975.17`), and confirmed that none of the three candidates acted as a strong safe haven during 2018–2023.  
  Overall, the `ChMSV` `StateSpaceModel` and Bootstrap Particle Filter pipeline are fully validated in Python (`particles`), providing the foundation for parameter estimation."
* **Visual Cue:** Point to the two active summary bullets and the bottom Takeaway line.
* **Transition:** "Thank you for your attention—we are happy to take any questions."

---

### Slide 20 — Questions? *(~10s)*
* **Spoken Script:**  
  "Thank you! We are happy to take any questions."
* **Visual Cue:** Leave Slide 20 up during Q&A.

---

## Anticipated Prof. Chopin Q&A Prep

1. **Why did `ESSrmin = 1.0` reduce Monte Carlo variance compared to `ESSrmin = 0.5`?**  
   While adaptive resampling ($\gamma = 0.5$) often lowers variance in smooth regimes, in a 10-dimensional state space hit by sudden COVID crash outliers (e.g., `-12.77%` SPX, `-42.45%` ETH), skipping resampling just before a shock leaves uneven particle weights entering the outlier day. That amplifies weight collapse (`Min ESS = 2.1`). Resetting weights every step ($\gamma = 1.0$, matching R's `pomp::pfilter`) halved single-run `Std(log L)` from `9.11` to `4.16`.

2. **Why does Table 2 report `-9972.73 (0.65)` while your single-run Std is `4.16`?**  
   In R's `pomp`, `logmeanexp(..., se=TRUE)` aggregates $R = 12$ independent particle filter runs via $\log(\frac{1}{R}\sum_r \hat{L}_r)$ and computes a **jackknife standard error** on that multi-run aggregate—not a single-run standard deviation. On our real data with $\gamma = 1.0$ across 10 runs, our `logmeanexp` is `-9975.17` with jackknife `SE = 1.12` (and `SE of Mean = 1.32`), right in line with the paper's `SE = 0.65` over 12 runs.

3. **Why treat correlation as a hidden state instead of using a 30-day rolling correlation?**  
   On a single day $t$, we only observe one return vector ($n=1$), so instantaneous correlation is unobservable. Rolling windows suffer from lag, "ghost drops" when a crash exits the window 30 days later, and can fail positive definiteness when computed pairwise. Treating $\Sigma_t = \mathbf{L}_t\mathbf{V}_t\mathbf{L}_t^\top$ as a filtered latent state reacts rapidly to volatility/correlation shifts, filters single-day noise, quantifies uncertainty (90% bands), and guarantees a valid covariance matrix every day.

4. **Is the $LDL^\top$ Cholesky model sensitive to asset ordering?**  
   Yes—that is a classic structural property of Cholesky MSV models. Because $\Sigma_{t,ii} = \sum_{k=1}^i q_{ik,t}^2 \exp(h_{k,t})$ (with $q_{ii,t}=1$), asset $1$ (SPX) drives the conditional variance of assets $2, 3, 4$, whereas asset ordering slightly changes the implied marginal dynamics. Keeping Szczepocki's exact ordering `(SPX, Gold, Bitcoin, Ethereum)` makes our results directly comparable to Table 2 and naturally places the market index SPX first as the common factor.

5. **How does the Bootstrap Particle Filter scale if $p$ grows larger (e.g., 10 or 20 assets)?**  
   Latent dimension grows quadratically as $d_x = p(p+1)/2$ (for $p=10$, $d_x=55$; for $p=20$, $d_x=210$). Since the Bootstrap filter proposes blindly from the prior transition $f_t(x_t \mid x_{t-1})$ without looking at today's crash return $\mathbf{y}_t$, particle weights will collapse rapidly for large $p$ unless $N$ grows dramatically or we use Guided / Auxiliary Particle Filters (approximating the optimal proposal via Laplace/Taylor expansion) or tempering.

6. **Why did Ethereum's daily close timestamp matter in Table 1 replication?**  
   Crypto trades 24/7, so daily log-returns depend on when a data provider cuts the daily candle. Standard `00:00 UTC` feeds merge two sell-off waves on 12 March 2020 into a `-55.07%` return (distorting kurtosis to `12.62`). Stooq.pl is based in Poland and cuts daily candles at **Warsaw midnight (`00:00 CET/CEST`)**, which reproduces Szczepocki's `-42.86%` minimum (`-42.45%` in our file) and `8.59` kurtosis (`8.55`).
