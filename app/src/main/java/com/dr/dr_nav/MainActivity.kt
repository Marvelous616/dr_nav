package com.dr.dr_nav

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.dr.dr_nav.ui.ControlPanelFragment

class MainActivity : AppCompatActivity() {

    private val permissionReq = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fine   = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        val coarse = permissions[Manifest.permission.ACCESS_COARSE_LOCATION] ?: false
        if (!fine && !coarse) {
            Toast.makeText(this, "Location permission is required for DR.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            permissionReq.launch(arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ))
        }

        // Add the control panel fragment dynamically into the bottom sheet container
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.controlPanelContainer, ControlPanelFragment())
                .commit()
        }
    }

    companion object {
        init {
            System.loadLibrary("dr_nav")
        }
    }
}