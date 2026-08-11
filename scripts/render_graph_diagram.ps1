Add-Type -AssemblyName System.Drawing

$projectRoot = Split-Path -Parent $PSScriptRoot
$docsDir = Join-Path $projectRoot "docs"
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null
$outputPath = Join-Path $docsDir "graph_diagram.png"

$bitmap = New-Object System.Drawing.Bitmap 1600, 1000
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::White)

$titleFont = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
$labelFont = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Regular)
$smallFont = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Regular)
$boxPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(55, 65, 81), 2)
$linePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(75, 85, 99), 2)
$linePen.CustomEndCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(5, 7)
$textBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(31, 41, 55))
$labelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(75, 85, 99))

function Draw-Box([int]$x, [int]$y, [int]$width, [int]$height, [string]$title, [System.Drawing.Color]$fill) {
    $rect = New-Object System.Drawing.Rectangle($x, $y, $width, $height)
    $rectF = New-Object System.Drawing.RectangleF($x, $y, $width, $height)
    $brush = New-Object System.Drawing.SolidBrush($fill)
    $graphics.FillRectangle($brush, $rect)
    $graphics.DrawRectangle($boxPen, $rect)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $graphics.DrawString($title, $labelFont, $textBrush, $rectF, $format)
    $format.Dispose()
    $brush.Dispose()
}

function Draw-Arrow([int]$x1, [int]$y1, [int]$x2, [int]$y2, [string]$label = "") {
    $graphics.DrawLine($linePen, $x1, $y1, $x2, $y2)
    if ($label) {
        $midX = [int](($x1 + $x2) / 2)
        $midY = [int](($y1 + $y2) / 2)
        $graphics.FillRectangle([System.Drawing.Brushes]::White, $midX - 66, $midY - 10, 132, 20)
        $graphics.DrawString($label, $smallFont, $labelBrush, $midX - 62, $midY - 8)
    }
}

$graphics.DrawString("OrbitDesk Local Support Agent - LangGraph Workflow", $titleFont, $textBrush, 415, 18)

$triage = [System.Drawing.Color]::FromArgb(219, 234, 254)
$answerable = [System.Drawing.Color]::FromArgb(220, 252, 231)
$terminal = [System.Drawing.Color]::FromArgb(254, 242, 242)
$verify = [System.Drawing.Color]::FromArgb(254, 249, 195)

Draw-Box 650 80 300 70 "Triage" $triage
Draw-Box 105 260 250 70 "Clarification" $terminal
Draw-Box 390 260 250 70 "Escalation" $terminal
Draw-Box 985 260 250 70 "Safe response" $terminal
Draw-Box 650 260 300 70 "Retrieval" $answerable
Draw-Box 650 440 300 70 "Generation" $answerable
Draw-Box 650 620 300 70 "Verification" $verify
Draw-Box 1260 620 240 70 "END" $terminal
Draw-Box 105 810 250 70 "END" $terminal
Draw-Box 445 810 250 70 "END" $terminal
Draw-Box 985 810 250 70 "END" $terminal

Draw-Arrow 800 150 800 260 "answerable"
Draw-Arrow 690 150 230 260 "requires clarification"
Draw-Arrow 760 150 570 260 "requires escalation"
Draw-Arrow 900 150 1110 260 "out of scope"
Draw-Arrow 800 330 800 440
Draw-Arrow 800 510 800 620
Draw-Arrow 230 330 230 810
Draw-Arrow 570 330 570 810
Draw-Arrow 1110 330 1110 810
Draw-Arrow 950 655 1260 655 "pass / safe failure"
Draw-Arrow 650 655 500 655 "fail once: feedback + retry"
Draw-Arrow 500 655 500 475
Draw-Arrow 500 475 650 475

$graphics.DrawString("Deterministic guard before model triage | local embedding retrieval | one revision max | runtime schema validation", $smallFont, $labelBrush, 275, 935)

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$linePen.Dispose()
$boxPen.Dispose()
$titleFont.Dispose()
$labelFont.Dispose()
$smallFont.Dispose()
$textBrush.Dispose()
$labelBrush.Dispose()
$graphics.Dispose()
$bitmap.Dispose()

Write-Output "Created $outputPath"
