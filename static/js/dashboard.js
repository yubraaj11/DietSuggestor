document.addEventListener("DOMContentLoaded", function () {
    const currentPage = window.location.pathname;
  
    if (currentPage.includes("/dashboard")) {
      document.getElementById("nav-dashboard").classList.add("active");
    } else if (currentPage.includes("/suggest-meal")) {
      document.getElementById("nav-recommendation").classList.add("active");
    }
  });
  


document.addEventListener("DOMContentLoaded", function () {
    fetch("/dashboard/", {
        headers: { "X-Requested-With": "XMLHttpRequest" }
    })
    .then(response => response.json())
    .then(data => {
        const { meals, summary } = data;

        document.getElementById("nutrition-summary").innerHTML = `
            <p><strong>Total Calories:</strong> ${summary.total_calories} kcal</p>
            <p><strong>Total Protein:</strong> ${summary.total_protein} g</p>
            <p><strong>Total Carbohydrates:</strong> ${summary.total_carbohydrates} g</p>
            <p><strong>Total Fats:</strong> ${summary.total_fats} g</p>
            <p><strong>Total Fiber:</strong> ${summary.total_fiber} g</p>
        `;

        const tableBody = document.getElementById("meals-table-body");
        tableBody.innerHTML = ""; 

        meals.forEach(meal => {
            const row = document.createElement("tr");

            const dateCell = document.createElement("td");
            dateCell.textContent = meal.date; 
            row.appendChild(dateCell);

            const mealTypeCell = document.createElement("td");
            mealTypeCell.textContent = meal.meal_type;
            row.appendChild(mealTypeCell);

            const recommendedDietCell = document.createElement("td");
            recommendedDietCell.textContent = meal.recommended_diet;
            row.appendChild(recommendedDietCell);

            const caloriesCell = document.createElement("td");
            caloriesCell.textContent = meal.calories;
            row.appendChild(caloriesCell);

            const proteinCell = document.createElement("td");
            proteinCell.textContent = meal.protein;
            row.appendChild(proteinCell);

            const carbsCell = document.createElement("td");
            carbsCell.textContent = meal.carbohydrates;
            row.appendChild(carbsCell);

            const fatsCell = document.createElement("td");
            fatsCell.textContent = meal.fats;
            row.appendChild(fatsCell);

            const fiberCell = document.createElement("td");
            fiberCell.textContent = meal.fiber;
            row.appendChild(fiberCell);

            tableBody.appendChild(row);
        });

        const summedData = {
            calories: summary.total_calories,
            protein: summary.total_protein,
            carbohydrates: summary.total_carbohydrates,
            fats: summary.total_fats,
            fiber: summary.total_fiber
        };

        const ctx = document.getElementById("macroChart").getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Calories", "Protein", "Carbohydrates", "Fats", "Fiber"],
                datasets: [
                    {
                        label: "Total Calories",
                        data: [summedData.calories, 0, 0, 0, 0],
                        backgroundColor: "rgba(255, 0, 55, 0.6)"
                    },
                    {
                        label: "Total Protein",
                        data: [0, summedData.protein, 0, 0, 0],
                        backgroundColor: "rgba(0, 88, 146, 0.6)"
                    },
                    {
                        label: "Total Carbohydrates",
                        data: [0, 0, summedData.carbohydrates, 0, 0],
                        backgroundColor: "rgba(0, 185, 185, 0.6)"
                    },
                    {
                        label: "Total Fats",
                        data: [0, 0, 0, summedData.fats, 0],
                        backgroundColor: "rgba(175, 126, 0, 0.6)"
                    },
                    {
                        label: "Total Fiber",
                        data: [0, 0, 0, 0, summedData.fiber],
                        backgroundColor: "rgba(57, 0, 172, 0.6)"
                    }
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

    })
    .catch(error => console.error("Error loading data:", error));
});
