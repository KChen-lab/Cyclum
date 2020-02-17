plot.MclustDA <- function (x, what = c("scatterplot", "classification", "train&test", 
                                       "error"), newdata, newclass, dimens, symbols, colors, ...) 
{
  object <- x
  if (!inherits(object, "MclustDA")) 
    stop("object not of class \"MclustDA\"")
  data <- object$data
  if (object$d > 1) 
    dataNames <- colnames(data)
  else dataNames <- deparse(object$call$data)
  n <- nrow(data)
  p <- ncol(data)
  if (missing(newdata)) {
    newdata <- matrix(as.double(NA), 0, p)
  }
  else {
    newdata <- as.matrix(newdata)
  }
  if (ncol(newdata) != p) 
    stop("incompatible newdata dimensionality")
  if (missing(newclass)) {
    newclass <- vector(length = 0)
  }
  else {
    if (nrow(newdata) != length(newclass)) 
      stop("incompatible newdata and newclass")
  }
  models <- object$models
  M <- length(models)
  if (missing(dimens)) 
    dimens <- 1:p
  trainClass <- object$class
  nclass <- length(unique(trainClass))
  Data <- rbind(data, newdata)
  predClass <- predict(object, Data)$classification
  if (missing(symbols)) {
    if (M <= length(mclust.options("classPlotSymbols"))) {
      symbols <- mclust.options("classPlotSymbols")
    }
    else if (M <= 26) {
      symbols <- LETTERS
    }
  }
  if (length(symbols) == 1) 
    symbols <- rep(symbols, M)
  if (length(symbols) < M & !any(what == "train&test")) {
    warning("more symbols needed to show classification")
    symbols <- rep(16, M)
  }
  if (missing(colors)) {
    colors <- mclust.options("classPlotColors")
  }
  if (length(colors) == 1) 
    colors <- rep(colors, M)
  if (length(colors) < M & !any(what == "train&test")) {
    warning("more colors needed to show classification")
    colors <- rep("black", M)
  }
  what <- match.arg(what, several.ok = TRUE)
  oldpar <- par(no.readonly = TRUE)
  plot.MclustDA.scatterplot <- function(...) {
    if (length(dimens) == 1) {
      eval.points <- seq(min(data[, dimens]), max(data[, 
                                                       dimens]), length = 1000)
      d <- matrix(as.double(NA), length(eval.points), nclass)
      for (i in seq(nclass)) {
        par <- models[[i]]$parameters
        if (par$variance$d > 1) {
          par$d <- 1
          par$mean <- par$mean[dimens, , drop = FALSE]
          par$variance$sigmasq <- par$variance$sigma[dimens, 
                                                     dimens, ]
          par$variance$modelName <- if (par$variance$G == 
                                        1) 
            "X"
          else if (dim(par$variance$sigma)[3] > 1) 
            "V"
          else "E"
        }
        d[, i] <- dens(modelName = par$variance$modelName, 
                       data = eval.points, parameters = par)
      }
      matplot(eval.points, d, type = "l", lty = 1, col = colors[seq(nclass)], 
              xlab = dataNames[dimens], ylab = "Density")
      for (i in 1:dim(d)[2])
        polygon(x = c(eval.points[1], eval.points, eval.points[length(eval.points)]), y = c(0, d[, i], 0), border = NA, col = paste(colors[seq(nclass)][i], '40', sep=""))
      matlines(eval.points, d, type = "l", lty = 1, col = colors[seq(nclass)], 
               xlab = dataNames[dimens], ylab = "Density")
      
      for (i in 1:nclass) {
        I <- models[[i]]$observations
        Axis(side = 1, at = data[I, ], labels = FALSE, 
             lwd = 0, lwd.ticks = 0.5, col.ticks = colors[i], 
             tck = -0.1)
      }
    }
    scatellipses <- function(data, dimens, nclass, symbols, 
                             colors, ...) {
      m <- lapply(models, function(m) {
        m$parameters$mean <- array(m$parameters$mean[dimens, 
                                                     ], c(2, m$G))
        m$parameters$variance$sigma <- array(m$parameters$variance$sigma[dimens, 
                                                                         dimens, ], c(2, 2, m$G))
        m
      })
      plot(data[, dimens], type = "n", ...)
      for (l in 1:nclass) {
        I <- m[[l]]$observations
        points(data[I, dimens[1]], data[I, dimens[2]], 
               pch = symbols[l], col = colors[l])
        for (k in 1:(m[[l]]$G)) {
          mvn2plot(mu = m[[l]]$parameters$mean[, k], 
                   sigma = m[[l]]$parameters$variance$sigma[, 
                                                            , k], k = 15)
        }
      }
    }
    if (length(dimens) == 2) {
      scatellipses(data, dimens, nclass, symbols, colors, 
                   ...)
    }
    if (length(dimens) > 2) {
      gap <- 0.2
      on.exit(par(oldpar))
      par(mfrow = c(p, p), mar = rep(c(gap, gap/2), each = 2), 
          oma = c(4, 4, 4, 4))
      for (i in seq(p)) {
        for (j in seq(p)) {
          if (i == j) {
            plot(0, 0, type = "n", xlab = "", ylab = "", 
                 axes = FALSE)
            text(0, 0, dataNames[i], cex = 1.5, adj = 0.5)
            box()
          }
          else {
            scatellipses(data, c(j, i), nclass, symbols, 
                         colors, xaxt = "n", yaxt = "n")
          }
          if (i == 1 && (!(j%%2))) 
            axis(3)
          if (i == p && (j%%2)) 
            axis(1)
          if (j == 1 && (!(i%%2))) 
            axis(2)
          if (j == p && (i%%2)) 
            axis(4)
        }
      }
    }
  }
  plot.MclustDA.classification <- function(...) {
    if (nrow(newdata) == 0 & length(dimens) == 1) {
      mclust1Dplot(data = data[, dimens], what = "classification", 
                   classification = predClass[1:n], colors = colors[1:nclass], 
                   xlab = dataNames[dimens], main = FALSE)
      title("Training data: known classification", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) == 0 & length(dimens) == 2) {
      coordProj(data = data[, dimens], what = "classification", 
                classification = predClass[1:n], main = FALSE, 
                colors = colors[1:nclass], symbols = symbols[1:nclass])
      title("Training data: known classification", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) == 0 & length(dimens) > 2) {
      clPairs(data[, dimens], classification = predClass[1:n], 
              colors = colors[1:nclass], symbols = symbols[1:nclass], 
              gap = 0.2, cex.labels = 1.5, main = "Training data: known classification", 
              cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) > 0 & length(dimens) == 1) {
      mclust1Dplot(data = newdata[, dimens], what = "classification", 
                   classification = predClass[-(1:n)], main = FALSE, 
                   xlab = dataNames[dimens])
      title("Test data: MclustDA classification", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) > 0 & length(dimens) == 2) {
      coordProj(data = newdata[, dimens], what = "classification", 
                classification = predClass[-(1:n)], main = FALSE, 
                colors = colors[1:nclass], symbols = symbols[1:nclass])
      title("Test data: MclustDA classification", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) > 0 & length(dimens) > 2) {
      on.exit(par(oldpar))
      par(oma = c(0, 0, 10, 0))
      clPairs(data = newdata[, dimens], classification = predClass[-(1:n)], 
              colors = colors[1:nclass], symbols = symbols[1:nclass], 
              gap = 0.2, cex.labels = 1.5, main = "Test data: MclustDA classification", 
              cex.main = oldpar$cex.lab)
    }
  }
  plot.MclustDA.traintest <- function(...) {
    if (length(dimens) == 1) {
      cl <- c(rep("Train", nrow(data)), rep("Test", nrow(newdata)))
      mclust1Dplot(data = Data[, dimens], what = "classification", 
                   classification = cl, main = FALSE, xlab = dataNames[dimens], 
                   colors = c("black", "red"))
      title("Training and Test  data", cex.main = oldpar$cex.lab)
    }
    if (length(dimens) == 2) {
      cl <- c(rep("1", nrow(data)), rep("2", nrow(newdata)))
      coordProj(Data[, dimens], what = "classification", 
                classification = cl, main = FALSE, CEX = 0.8, 
                symbols = c(1, 3), colors = c("black", "red"))
      title("Training (o) and Test (+) data", cex.main = oldpar$cex.lab)
    }
    if (length(dimens) > 2) {
      cl <- c(rep("1", nrow(data)), rep("2", nrow(newdata)))
      clPairs(Data[, dimens], classification = cl, symbols = c(1, 
                                                               3), colors = c("black", "red"), gap = 0.2, cex.labels = 1.3, 
              CEX = 0.8, main = "Training (o) and Test (+) data", 
              cex.main = oldpar$cex.lab)
    }
  }
  plot.MclustDA.error <- function(...) {
    if (nrow(newdata) != length(newclass)) 
      stop("incompatible newdata and newclass")
    if (nrow(newdata) == 0 & length(dimens) == 1) {
      mclust1Dplot(data = data[, dimens], what = "errors", 
                   classification = predClass[1:n], truth = trainClass, 
                   xlab = dataNames[dimens], main = FALSE)
      title("Train Error", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) == 0 & length(dimens) > 1) {
      coordProj(data = data[, dimens[1:2]], what = "errors", 
                classification = predClass[1:n], truth = trainClass, 
                main = FALSE)
      title("Train Error", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) > 0 & length(dimens) == 1) {
      mclust1Dplot(data = newdata[, dimens], what = "errors", 
                   classification = predClass[-(1:n)], truth = newclass, 
                   xlab = dataNames[dimens], main = FALSE)
      title("Test Error", cex.main = oldpar$cex.lab)
    }
    if (nrow(newdata) > 0 & length(dimens) > 1) {
      coordProj(data = newdata[, dimens[1:2]], what = "errors", 
                classification = predClass[-(1:n)], truth = newclass, 
                main = FALSE)
      title("Test Error", cex.main = oldpar$cex.lab)
    }
  }
  if (interactive() & length(what) > 1) {
    title <- "Model-based discriminant analysis plots:"
    choice <- menu(what, graphics = FALSE, title = title)
    while (choice != 0) {
      if (what[choice] == "scatterplot") 
        plot.MclustDA.scatterplot(...)
      if (what[choice] == "classification") 
        plot.MclustDA.classification(...)
      if (what[choice] == "train&test") 
        plot.MclustDA.traintest(...)
      if (what[choice] == "error") 
        plot.MclustDA.error(...)
      choice <- menu(what, graphics = FALSE, title = title)
    }
  }
  else {
    if (any(what == "scatterplot")) 
      plot.MclustDA.scatterplot(...)
    if (any(what == "classification")) 
      plot.MclustDA.classification(...)
    if (any(what == "train&test")) 
      plot.MclustDA.traintest(...)
    if (any(what == "error")) 
      plot.MclustDA.error(...)
  }
  invisible()
}
