package example

import scalation.mathstat.MatrixD
import scalation.modeling.{RidgeRegression, SymRidgeRegression}
import scala.collection.mutable.LinkedHashSet

@main def SymRidge_ParkinsonsMotor(): Unit =
  val xy = MatrixD.load("outputs/cleaned_for_scala/parkinsons_motor_xy.csv", 1, 0)
  val m  = xy.dim
  val n  = xy.dim2

  val x  = xy(0 until m, 0 until n-1)
  val y  = xy(0 until m, n-1)

  val fname: Array[String] = null
  val powers: LinkedHashSet[Double] = LinkedHashSet(2.0)

  val mod: RidgeRegression =
    SymRidgeRegression(x, y, fname, powers, cross = true, cross3 = false, RidgeRegression.hp)

  mod.trainNtest()()
  //println(mod.summary())
