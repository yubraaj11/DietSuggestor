// document.querySelector("form").addEventListener("submit", async (e) => {
//   e.preventDefault();

//   const username = document.getElementById("username").value;
//   const password = document.getElementById("password").value;

//   const payload = {
//     username: username,
//     password: password,
//   };

//   try {
//     const response = await fetch("https://jsonplaceholder.typicode.com/posts", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify(payload),
//     });

//     const data = await response.json();
//     console.log("Response:", data);
//   } catch (error) {
//     console.error("Error:", error);
//   }
// });
