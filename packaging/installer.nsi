Unicode True
!include "MUI2.nsh"
Name "Possibility Lab"
OutFile "..\releases\PossibilityLab-Windows-x64-Setup.exe"
InstallDir "$LOCALAPPDATA\Programs\PossibilityLab"
RequestExecutionLevel user
SetCompressor /SOLID lzma
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Install Possibility Lab"
!define MUI_WELCOMEPAGE_TEXT "An offline Parallel Possibility Engine demo. Python, the desktop runtime and all trained models are included. No downloads or setup are needed after installation."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\PossibilityLab.exe"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Section "Install"
 SetOutPath "$INSTDIR"
 File /r "dist-windows\PossibilityLab\*"
 CreateDirectory "$SMPROGRAMS\Possibility Lab"
 CreateShortcut "$SMPROGRAMS\Possibility Lab\Possibility Lab.lnk" "$INSTDIR\PossibilityLab.exe"
 CreateShortcut "$DESKTOP\Possibility Lab.lnk" "$INSTDIR\PossibilityLab.exe"
 WriteUninstaller "$INSTDIR\Uninstall.exe"
 WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab" "DisplayName" "Possibility Lab"
 WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab" "UninstallString" '$"$INSTDIR\Uninstall.exe$"'
 WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab" "DisplayVersion" "1.0.0"
 WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab" "NoModify" 1
 WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab" "NoRepair" 1
SectionEnd
Section "Uninstall"
 Delete "$DESKTOP\Possibility Lab.lnk"
 Delete "$SMPROGRAMS\Possibility Lab\Possibility Lab.lnk"
 RMDir "$SMPROGRAMS\Possibility Lab"
 RMDir /r "$INSTDIR"
 DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\PossibilityLab"
SectionEnd
