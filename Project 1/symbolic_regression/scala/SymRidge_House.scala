package example

import scalation.mathstat.MatrixD
import scalation.modeling.{RidgeRegression, SymRidgeRegression}
import scala.collection.mutable.LinkedHashSet

@main def SymRidge_House(): Unit =
  val xy = MatrixD.load("outputs/cleaned_for_scala/house_xy.csv", 1, 0)
  val m  = xy.dim
  val n  = xy.dim2

  val x  = xy(0 until m, 0 until n-1)
  val y  = xy(0 until m, n-1)

  val fname = Array(
    "Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built","Lot_Size",
    "Garage_Size","Neighborhood_Quality"
  )

  val powers: LinkedHashSet[Double] = LinkedHashSet(2.0, 0.5)

  val mod: RidgeRegression =
    SymRidgeRegression(x, y, fname, powers, cross = true, cross3 = false, RidgeRegression.hp)

  mod.trainNtest()()
  //println(mod.summary())
