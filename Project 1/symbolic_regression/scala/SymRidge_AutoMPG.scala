package example

import scalation.mathstat.{MatrixD}
import scalation.modeling.{RidgeRegression, SymRidgeRegression}
import scala.collection.mutable.LinkedHashSet

@main def SymRidge_AutoMPG(): Unit =
  // Option B: absolute path to Project 1 CSV
  val xy = MatrixD.load("outputs/cleaned_for_scala/autompg_xy.csv", 1, 0)


  val n  = xy.dim2

  // Avoid `?` row selector by using explicit row ranges
  val x  = xy(0 until xy.dim, 0 until n-1)     // features
  val y  = xy(0 until xy.dim, n-1)             // response (VectorD)

  val fname = Array(
    "cylinders","displacement","horsepower","weight","acceleration","model_year",
    "origin_1","origin_2","origin_3"
  )

  val powers = LinkedHashSet(2.0, 0.5)

  val mod: RidgeRegression =
    SymRidgeRegression(x, y, fname, powers, cross = true, cross3 = false, RidgeRegression.hp)

  mod.trainNtest()()
  //println(mod.summary())
