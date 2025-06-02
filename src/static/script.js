const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImg = document.getElementById('preview-img');
const resultDiv = document.getElementById('result');
const fileLabel = document.getElementById('file-label');
const analysisTypeSelect = document.getElementById('analysis-type');
const predictionsDiv = document.getElementById('predictions');

fileInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        fileLabel.textContent = file.name;
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewContainer.style.display = 'block';
        }
        reader.readAsDataURL(file);
    } else {
        fileLabel.textContent = 'Chọn ảnh';
        previewContainer.style.display = 'none';
    }
});

form.addEventListener('submit', function(e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('analysis_type', analysisTypeSelect.value); 

    resultDiv.textContent = 'Đang xử lý...';
    predictionsDiv.textContent = '';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultDiv.textContent = data.error;
            resultDiv.style.color = 'red';
        } else {
            resultDiv.textContent = data.result;
            resultDiv.style.color = '#0078d7';
           
        }
    })
    .catch(() => {
        resultDiv.textContent = 'Có lỗi xảy ra khi xử lý ảnh.';
        resultDiv.style.color = 'red';
    });
});
