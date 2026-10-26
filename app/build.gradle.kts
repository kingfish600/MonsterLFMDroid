import java.io.File

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.prismml.bonsaidroid"
    compileSdk = 34

    defaultConfig {
        // applicationId differs so both apps coexist; namespace is kept from the
        // BonsaiDroid scaffolding so the reused Kotlin needs no renaming.
        applicationId = "io.github.kingfish600.monsterqwendroid"
        minSdk = 31              // Android 12+; RedMagic 10S Pro ships 15/16
        targetSdk = 34
        versionCode = 2
        versionName = "0.1.1-alpha"

        ndk {
            // Snapdragon 8 Elite is arm64 only.
            abiFilters += listOf("arm64-v8a")
        }
    }

    signingConfigs {
        getByName("debug") {
            // Sandbox-friendly: keystore lives in the repo (password: android).
            storeFile = rootProject.file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
            // AGP disables v1 (JAR) signing whenever minSdk >= 24, which leaves the
            // APK signed with v2 only - no META-INF signature files at all. Android
            // 7+ accepts that, but some OEM installer paths (OxygenOS/ColorOS, and
            // third-party file managers) read the JAR manifest and fail with
            // "Parsing failed: APK contains no signature files". Enable all three
            // schemes: v1 for those installers, v2/v3 for modern Android.
            // See issue: OnePlus 13 / Android 16 could not install 0.1.0-alpha.
            enableV1Signing = true
            enableV2Signing = true
            enableV3Signing = true
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug") // replace with your keystore
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-opt-in=kotlin.RequiresOptIn", "-Xjvm-default=all")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        // CRITICAL: extract native libs to disk so the bundled llama-server
        // executable in nativeLibraryDir keeps its execute bit.
        jniLibs {
            useLegacyPackaging = true
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    sourceSets {
        getByName("main") {
            // The engine ships prebuilt in src/main/jniLibs (see jniLibs/arm64-v8a).
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.service)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons)
    implementation(libs.androidx.navigation.compose)
    debugImplementation(libs.androidx.compose.ui.tooling)

    implementation(libs.androidx.datastore.preferences)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.okhttp)
    implementation(libs.okhttp.sse)
}
