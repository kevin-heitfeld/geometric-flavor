@echo off
REM Compile Paper 3: Dark Energy Manuscript
REM Usage: compile.bat

echo ========================================
echo Compiling Paper 3: Dark Energy
echo ========================================
echo.

cd manuscript_dark_energy

echo [1/4] First LaTeX pass...
pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1
if errorlevel 1 (
    echo ERROR in first pass! Check compile.log
    type compile.log | findstr /C:"!" 
    goto :error
)

echo [2/4] Building bibliography...
bibtex main >> compile.log 2>&1
if errorlevel 1 (
    echo WARNING: BibTeX errors (may be OK if no citations yet^)
)

echo [3/4] Second LaTeX pass...
pdflatex -interaction=nonstopmode main.tex >> compile.log 2>&1
if errorlevel 1 (
    echo ERROR in second pass! Check compile.log
    goto :error
)

echo [4/4] Final LaTeX pass...
pdflatex -interaction=nonstopmode main.tex >> compile.log 2>&1
if errorlevel 1 (
    echo ERROR in final pass! Check compile.log
    goto :error
)

echo.
echo ========================================
echo SUCCESS! PDF generated: main.pdf
echo ========================================
echo.
echo Opening PDF...
start main.pdf

cd ..
goto :end

:error
cd ..
echo.
echo ========================================
echo COMPILATION FAILED
echo ========================================
exit /b 1

:end
