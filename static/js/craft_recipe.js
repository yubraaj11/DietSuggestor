// craft-recipe.js - Simplified API handling

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const ingredientsForm = document.getElementById('ingredients-form');
    const ingredientsTextarea = document.getElementById('ingredients');
    const generateRecipeBtn = document.getElementById('generate-recipe-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const recipeContent = document.getElementById('recipe-content');
    const recipeName = document.getElementById('recipe-name');
    const recipeSteps = document.getElementById('recipe-steps');
    
    // Initially hide the loading indicator
    loadingIndicator.style.display = 'none';
    
    // Handle form submission
    ingredientsForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Get ingredients text and validate
        const ingredientsText = ingredientsTextarea.value.trim();
        if (!ingredientsText) {
            alert('Please enter at least one ingredient');
            return;
        }
        
        // Convert comma-separated ingredients to array
        const ingredients = ingredientsText
            .split(',')
            .map(item => item.trim())
            .filter(item => item.length > 0);
        
        // Show loading state
        loadingIndicator.style.display = 'block';
        recipeContent.style.display = 'none';
        generateRecipeBtn.disabled = true;
        
        // Make API request to generate recipe
        fetch('/craft-recipe/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ingredients: ingredients
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.detail || 'Failed to generate recipe');
                });
            }
            return response.json();
        })
        .then(data => {
            // Handle successful response
            recipeName.textContent = data.dish_name;
            
            // Clear previous recipe steps
            recipeSteps.innerHTML = '';
            
            // Create ordered list for steps
            const stepsList = document.createElement('ol');
            
            // Add each step as a list item
            data.steps.forEach(step => {
                const listItem = document.createElement('li');
                listItem.textContent = step;
                stepsList.appendChild(listItem);
            });
            
            // Add list to recipe steps container
            recipeSteps.appendChild(stepsList);
            
            // Hide loading, show content
            loadingIndicator.style.display = 'none';
            recipeContent.style.display = 'block';
            generateRecipeBtn.disabled = false;
            
            // Scroll to result
            document.getElementById('recipe-result').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Handle error
            recipeName.textContent = 'Error';
            recipeSteps.innerHTML = `<p class="error-message">Sorry, we couldn't generate a recipe: ${error.message}</p>`;
            
            // Hide loading, show content with error
            loadingIndicator.style.display = 'none';
            recipeContent.style.display = 'block';
            generateRecipeBtn.disabled = false;
            
            console.error('Recipe generation error:', error);
        });
    });
});