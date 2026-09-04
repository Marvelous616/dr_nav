package com.dr.dr_nav.map

/**
 * No-op map matcher — returns the input position unchanged.
 * Used as the baseline when no road graph is loaded, or map matching is disabled.
 */
class PassthroughMatcher : MapMatcher {
    override val name = "Passthrough"

    override fun match(
        lat       : Double,
        lon       : Double,
        headingRad: Float,
        speedMs   : Float,
        posStdM   : Float,
    ) = MatchResult(
        lat         = lat,
        lon         = lon,
        snapped     = false,
        roadBearing = headingRad,
        confidence  = 1f,
        mode        = MatchMode.PASSTHROUGH,
    )
}
