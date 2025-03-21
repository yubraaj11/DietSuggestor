document.addEventListener("DOMContentLoaded", function () {
    const currentPage = window.location.pathname;
  
    if (currentPage.includes("/dashboard")) {
      document.getElementById("nav-dashboard").classList.add("active");
    } else if (currentPage.includes("/suggest-meal")) {
      document.getElementById("nav-recommendation").classList.add("active");
    }
  });
  

document.getElementById("diet-form").addEventListener("submit", function (event) {
    event.preventDefault();

    document.getElementById("loader").style.display = "block";

    document.getElementById("recommendation").style.display = "none";

    const formData = new FormData(this);

    fetch("/suggest-meal/", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        const mealType = data.meal_type;
        const recommendedDiet = data.recommended_diet;
        const imagePath = data.image_path;
        const currentTime = data.current_time;

        document.getElementById("meal-name").textContent = mealType;
        document.getElementById("recommendation-image").src = imagePath;
        document.getElementById("meal-info").textContent = `${recommendedDiet}`;

        document.getElementById("recommendation").style.display = "block";

        document.getElementById("log-meal-button").setAttribute("data-meal-type", mealType);
        document.getElementById("log-meal-button").setAttribute("data-recommended-diet", recommendedDiet);
        document.getElementById("log-meal-button").style.display = "block";
    })
    .catch(error => console.error("Error:", error))
    .finally(() => {
        document.getElementById("loader").style.display = "none";
    });
});

document.getElementById("log-meal-button").addEventListener("click", function () {
    const mealType = this.getAttribute("data-meal-type");
    const recommendedDiet = this.getAttribute("data-recommended-diet");

    fetch("/log-meal/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            meal_type: mealType,
            recommended_diet: recommendedDiet,
        }),
    })
    .then(response => response.json())
    .then(data => {
        window.location.href = "/dashboard";
    })
    .catch(error => console.error("Error:", error));
});