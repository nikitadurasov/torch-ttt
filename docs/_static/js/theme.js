document.addEventListener("DOMContentLoaded", function () {
    // Find the first Jupyter Notebook download link
    let notebookLink = document.querySelector("div.sphx-glr-download-jupyter a.reference.download.internal");

    if (notebookLink) {
        // Get the full href (e.g., "../_downloads/25c1caf8.../01_mnist_ttt.ipynb")
        let ipynbPath = notebookLink.getAttribute("href");

        // Extract only the file name (e.g., "01_mnist_ttt.ipynb")
        let notebookFileName = ipynbPath.split("/").pop();

        // Final Colab URL pointing to the correct GitHub location
        let colabURL = `https://colab.research.google.com/github/torch-ttt/torch-ttt.github.io/blob/main/auto_examples/downloads/${notebookFileName}`;

        // Create the Colab button
        let colabButton = document.createElement("a");
        colabButton.href = colabURL;
        colabButton.target = "_blank";
        colabButton.innerHTML = '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">';
        colabButton.style.display = "block";
        colabButton.style.marginTop = "10px";

        // Insert below the title (h1)
        let title = document.querySelector("h1");
        if (title) {
            title.insertAdjacentElement("afterend", colabButton);
        }
    }
});
