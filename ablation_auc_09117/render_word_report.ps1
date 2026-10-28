param([Parameter(Mandatory=$true)][string]$DocumentPath)
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
$ownedApplication = $false
try {
    $word = New-Object -ComObject Word.Application
    if ($word.Documents.Count -ne 0) { throw 'The Word instance has other documents; refusing to modify it.' }
    $ownedApplication = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($DocumentPath, $false, $false, $false)
    $document.Repaginate()
    $null = $document.Fields.Update()
    foreach ($section in $document.Sections) {
        foreach ($footer in $section.Footers) { $null = $footer.Range.Fields.Update() }
    }
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $pageData = @()
    for ($index = 1; $index -le $pageCount; $index++) {
        $start = $document.GoTo(1, 1, $index).Start
        $end = if ($index -lt $pageCount) { $document.GoTo(1, 1, ($index + 1)).Start } else { $document.Content.End }
        $pageText = $document.Range($start, $end).Text
        $pageData += [ordered]@{page=$index; characters=$pageText.Length; text=$pageText}
    }
    $pictureData = @()
    foreach ($picture in $document.InlineShapes) {
        $pictureData += [ordered]@{page=$picture.Range.Information(3); width_pt=$picture.Width; height_pt=$picture.Height}
    }
    $document.Save()
    $pdfPath = [System.IO.Path]::ChangeExtension($DocumentPath, '.pdf')
    $document.ExportAsFixedFormat($pdfPath, 17)
    $layout = [ordered]@{renderer='Microsoft Word'; pages=$pageCount; tables=$document.Tables.Count; images=$document.InlineShapes.Count; pictures=$pictureData; page_content=$pageData; pdf=$pdfPath}
    $layoutPath = [System.IO.Path]::ChangeExtension($DocumentPath, '.layout.json')
    $layout | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $layoutPath -Encoding UTF8
    [ordered]@{pages=$pageCount; tables=$document.Tables.Count; images=$document.InlineShapes.Count; pdf=$pdfPath; layout=$layoutPath} | ConvertTo-Json -Compress
}
finally {
    $saveOptions = 0
    if ($null -ne $document) { $document.Close([ref]$saveOptions); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($ownedApplication -and $null -ne $word) { $word.Quit([ref]$saveOptions) }
    if ($null -ne $word) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
