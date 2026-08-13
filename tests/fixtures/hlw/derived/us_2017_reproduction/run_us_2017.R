# Driver script (not part of the official HLW release) that runs the
# genuine HLW (2017) three-stage US estimation on data we already have,
# bypassing prepare.rstar.data.us.R's FRED auto-fetch. Mirrors run.hlw.R's
# US section exactly (same call signature, same constraints), and
# additionally extracts the smoothed (two-sided) series that run.hlw.R
# computes internally but never writes to CSV.
#
# Run from this directory: Rscript run_us_2017.R
# Reads:  inputData/rstar.data.us.csv
# Writes: output/us_2017_{parameters,one_sided,smoothed}.csv
code.dir <- "../../HLW_2017_Code"

library(tis)
library(mFilter)
library(nloptr)

source(file.path(code.dir, "calculate.covariance.R"))
source(file.path(code.dir, "format.output.R"))
source(file.path(code.dir, "kalman.log.likelihood.R"))
source(file.path(code.dir, "kalman.standard.errors.R"))
source(file.path(code.dir, "kalman.states.R"))
source(file.path(code.dir, "kalman.states.wrapper.R"))
source(file.path(code.dir, "log.likelihood.wrapper.R"))
source(file.path(code.dir, "median.unbiased.estimator.stage1.R"))
source(file.path(code.dir, "median.unbiased.estimator.stage2.R"))
source(file.path(code.dir, "rstar.stage1.R"))
source(file.path(code.dir, "rstar.stage2.R"))
source(file.path(code.dir, "rstar.stage3.R"))
source(file.path(code.dir, "run.hlw.estimation.R"))
source(file.path(code.dir, "unpack.parameters.stage1.R"))
source(file.path(code.dir, "unpack.parameters.stage2.R"))
source(file.path(code.dir, "unpack.parameters.stage3.R"))
source(file.path(code.dir, "utilities.R"))

a3.constraint <- -0.0025
b2.constraint <- 0.025

sample.start <- c(1961, 1)
sample.end   <- c(2019, 2)
data.start   <- shiftQuarter(sample.start, -4)

g.pot.start.index <- 1 + ti(shiftQuarter(sample.start, -3), 'quarterly') - ti(data.start, 'quarterly')

us.data <- read.table("inputData/rstar.data.us.csv",
                       sep = ',', na.strings = ".", header = TRUE, stringsAsFactors = FALSE)

us.log.output             <- us.data$gdp.log
us.inflation               <- us.data$inflation
us.inflation.expectations <- us.data$inflation.expectations
us.nominal.interest.rate  <- us.data$interest
us.real.interest.rate     <- us.nominal.interest.rate - us.inflation.expectations

cat("Data rows:", length(us.log.output), "(expect 238 = T+4 with T=234)\n")

# run.se=FALSE: skip the 5000-iteration Monte Carlo standard-error procedure.
# Not needed for a point-estimate G5a oracle; documented simplification.
us.estimation <- run.hlw.estimation(us.log.output, us.inflation, us.real.interest.rate, us.nominal.interest.rate,
                                     a3.constraint = a3.constraint, b2.constraint = b2.constraint, run.se = FALSE)

out3 <- us.estimation$out.stage3

cat("\n=== Stage 3 theta (a_y1, a_y2, a_r, b_pi, b_y, sigma_ytilde, sigma_pi, sigma_ystar) ===\n")
print(out3$theta)
cat("lambda.g:", us.estimation$lambda.g, "  lambda.z:", us.estimation$lambda.z, "\n")
cat("log-likelihood:", out3$log.likelihood, "\n")

dir.create("output", showWarnings = FALSE)

# Save parameters. sigma_z is |lambda_z * sigma_ytilde / a_r| -- unpack.parameters.stage3.R
# squares this term when building Q, so its sign is an artifact of a_r's sign, not meaningful.
params.out <- data.frame(
  parameter = c("a_y1","a_y2","a_r","b_pi","b_y","sigma_ytilde","sigma_pi","sigma_ystar",
                "lambda_g","lambda_z","sigma_g","sigma_z","log_likelihood"),
  value = c(out3$theta,
            us.estimation$lambda.g, us.estimation$lambda.z,
            us.estimation$lambda.g * out3$theta[8],
            abs(us.estimation$lambda.z * out3$theta[6] / out3$theta[3]),
            out3$log.likelihood)
)
write.csv(params.out, "output/us_2017_parameters.csv", row.names = FALSE)

one.sided <- data.frame(
  rstar = out3$rstar.filtered,
  g = out3$trend.filtered,
  z = out3$z.filtered,
  output_gap = out3$output.gap.filtered
)
write.csv(one.sided, "output/us_2017_one_sided.csv", row.names = FALSE)

two.sided <- data.frame(
  rstar = out3$rstar.smoothed,
  g = out3$trend.smoothed,
  z = out3$z.smoothed,
  output_gap = out3$output.gap.smoothed
)
write.csv(two.sided, "output/us_2017_smoothed.csv", row.names = FALSE)

cat("\nRows in one-sided/smoothed output:", nrow(one.sided), "(expect T=234, 1961Q1-2019Q2)\n")
cat("DONE\n")
