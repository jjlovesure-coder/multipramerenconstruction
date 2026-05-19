param(
    [Parameter(Position = 0)]
    [ValidateSet("summary", "recent", "changed", "staged", "caches", "push-command")]
    [string]$Command = "summary"
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([string[]]$GitArgs)
    & git @GitArgs
}

switch ($Command) {
    "summary" {
        Write-Host "== Branch =="
        Invoke-Git -GitArgs @("branch", "--show-current")
        Write-Host ""
        Write-Host "== Status =="
        Invoke-Git -GitArgs @("status", "-sb")
        Write-Host ""
        Write-Host "== Recent commits =="
        Invoke-Git -GitArgs @("log", "--oneline", "--decorate", "--graph", "-8")
    }
    "recent" {
        Invoke-Git -GitArgs @("log", "--oneline", "--decorate", "--graph", "-20")
    }
    "changed" {
        Invoke-Git -GitArgs @("diff", "--stat")
    }
    "staged" {
        Invoke-Git -GitArgs @("diff", "--cached", "--stat")
    }
    "caches" {
        Invoke-Git -GitArgs @("ls-files", "*__pycache__*")
    }
    "push-command" {
        Write-Host "git -c http.proxy= -c https.proxy= push origin Richard"
    }
}
