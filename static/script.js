document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();
  
    // Fetch input values
    var formData = new FormData(document.getElementById("predictionForm"));
  
    // Make AJAX request to Flask backend
    fetch("/", {
      method: "POST",
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // Display prediction result
      var predictionResult = document.getElementById("predictionResult");
      var classification = data.classification === 1 ? 'CKD' : 'Non-CKD';
      var ckdStage = data.ckd_stage;
      var diagnosis = data.diagnosis;
  
      // Determine message based on CKD stage
      var message = "";
      if (ckdStage <= 2) {
        message = "You are safe! You don't have CKD.";
      } else {
        message = "Oops! You have CKD.";
      }
  
      predictionResult.innerHTML = "<strong>Classification:</strong> " + classification + "<br>" +
                                   "<strong>CKD Stage:</strong> " + ckdStage + "<br>" +
                                   "<strong>Diagnosis:</strong> " + diagnosis + "<br>" +
                                   "<strong>Message:</strong> " + message;
    })
    .catch(error => {
      console.error("Error:", error);
    });
  });