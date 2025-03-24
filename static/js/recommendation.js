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
        const mealOptions = data.meal_options;
        const currentTime = data.current_time;
        
        const recommendationSection = document.getElementById("recommendation");
        recommendationSection.innerHTML = `
            <h2>Suggested ${mealType.charAt(0).toUpperCase() + mealType.slice(1)} Options (${currentTime})</h2>
            <div id="meal-options-container" class="meal-options-container"></div>
            <button id="log-meal-button" style="display: none;" class="log-meal-button">Log This Meal</button>
        `;
        
        const mealOptionsContainer = document.getElementById("meal-options-container");
        
        mealOptions.forEach(option => {
            const mealCard = document.createElement("div");
            mealCard.className = "meal-card";
            mealCard.innerHTML = `
                <img src="${option.image_path}" alt="${option.meal_name}" class="meal-image">
                <h3>${option.meal_name}</h3>
                <button class="select-meal-btn" data-meal-name="${option.meal_name}">Select</button>
            `;
            mealOptionsContainer.appendChild(mealCard);
        });
        
        document.querySelectorAll(".select-meal-btn").forEach(button => {
            button.addEventListener("click", function() {
                const selectedMealName = this.getAttribute("data-meal-name");
                
                const logMealButton = document.getElementById("log-meal-button");
                logMealButton.setAttribute("data-meal-type", mealType);
                logMealButton.setAttribute("data-recommended-diet", selectedMealName);
                logMealButton.style.display = "block";
                
                document.querySelectorAll(".meal-card").forEach(card => {
                    card.classList.remove("selected");
                });
                this.closest(".meal-card").classList.add("selected");
            });
        });
        
        recommendationSection.style.display = "block";
    })
    .catch(error => console.error("Error:", error))
    .finally(() => {
        document.getElementById("loader").style.display = "none";
    });
});

document.addEventListener("click", function(event) {
    if (event.target && event.target.id === "log-meal-button") {
        document.getElementById("loader").style.display = "block";
        
        const mealType = event.target.getAttribute("data-meal-type");
        const recommendedDiet = event.target.getAttribute("data-recommended-diet");
        
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
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("loader").style.display = "none";
        });
    }
});