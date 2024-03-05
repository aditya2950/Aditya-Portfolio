<?php
// Database connection parameters
$hostname = "localhost";
$username = "root";
$password = "123456";
$database = "adi";

// Connect to MySQL database
$connection = mysqli_connect($hostname, $username, $password, $database);

// Check connection
if (!$connection) {
    die("Connection failed: " . mysqli_connect_error());
}


$name=$_POST['name'];
$email=$_POST['email'];
$message=$_POST['message'];


$sql = "INSERT INTO aditya (fullname, email, message) VALUES ('$name', '$email', '$message')";

if (mysqli_query($connection, $sql)) {
    echo "Record inserted successfully";
} else {
    echo "Error: " . $sql . "<br>" . mysqli_error($connection);
}



// Close database connection
mysqli_close($connection);

?>
