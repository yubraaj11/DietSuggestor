// document.querySelector("form").addEventListener("submit", function (e) {
//   e.preventDefault(); // Prevent the default form submission

//   // Collect form data
//   const formData = {
//     fullName: document.querySelector("input[type='text']").value,
//     email: document.querySelector("input[type='email']").value,
//     password: document.querySelector("input[type='password']").value,
//     confirmPassword: document.querySelectorAll("input[type='password']")[1]
//       .value,
//     age: document.querySelector("input[type='number']").value,
//     gender: document.querySelector("select").value,
//     height: document.querySelectorAll("input[type='number']")[1].value,
//     weight: document.querySelectorAll("input[type='number']")[2].value,
//   };

//   // Send the form data as a JSON object (example using fetch)
//   fetch("YOUR_API_URL_HERE", {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//     },
//     body: JSON.stringify(formData),
//   })
//     .then((response) => response.json())
//     .then((data) => {
//       console.log("Success:", data);
//       // Redirect or display success message
//     })
//     .catch((error) => {
//       console.error("Error:", error);
//     });
// });
