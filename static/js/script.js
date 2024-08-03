function submitForm() {
    document.getElementById('stockForm').submit();
}

function toggleStockData() {
    var stockDataSection = document.getElementById('stockDataSection');
    var checkbox = document.getElementById('toggleStockData');
    if (checkbox.checked) {
        stockDataSection.style.display = 'block';
    } else {
        stockDataSection.style.display = 'none';
    }
}
