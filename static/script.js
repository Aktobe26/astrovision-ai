let video = null;
let canvas = null;
let stream = null;

function startCamera() {
    video = document.getElementById("video");
    canvas = document.getElementById("canvas");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(s) {
            stream = s;
            video.srcObject = stream;
            document.getElementById("camera-container").style.display = "block";
        })
        .catch(function(err) {
            alert("Не удалось получить доступ к камере.");
        });
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0);

    canvas.toBlob(function(blob) {
        let formData = new FormData();
        formData.append("file", blob, "camera.jpg");

        fetch("/upload/", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            document.open();
            document.write(html);
            document.close();
        });
    }, "image/jpeg");

    stream.getTracks().forEach(track => track.stop());
}