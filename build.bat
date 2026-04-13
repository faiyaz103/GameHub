@echo off
rem Build script for gamehub project
rem Uses Visual Studio 2022 Community MSBuild path automatically

set MSBUILD="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"

if exist %MSBUILD% (
    echo Building GameHub Project...
    %MSBUILD% gamehub.sln /p:Configuration=Debug /p:Platform=x64
    echo.
    if %errorlevel% == 0 (
        echo Build Successful! You can run the executable from:
        echo .\x64\Debug\gamehub.exe
    ) else (
        echo Build Failed!
    )
) else (
    echo MSBuild not found at expected location.
    echo Make sure Visual Studio 2022 Community is installed.
)
pause
