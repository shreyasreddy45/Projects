#!/bin/bash

# Function: Add a new car
add_car() {
  model=$(dialog --inputbox "Enter Car Model:" 8 40 3>&1 1>&2 2>&3)
  brand=$(dialog --inputbox "Enter Car Brand:" 8 40 3>&1 1>&2 2>&3)
  year=$(dialog --inputbox "Enter Car Year:" 8 40 3>&1 1>&2 2>&3)
  rate=$(dialog --inputbox "Enter Daily Rental Rate:" 8 40 3>&1 1>&2 2>&3)
  echo "$model,$brand,$year,$rate,Available" >> cars.txt
  dialog --msgbox "Car added successfully!" 6 20
}

# Function: View all cars (without status)
view_cars() {
  car_list=$(awk -F"," '{printf "Model: %s, Brand: %s, Year: %s\n", $1, $2, $3}' cars.txt)
  dialog --title "Car List" --msgbox "$car_list" 20 70
}

# Function: Add a new customer
add_customer() {
  name=$(dialog --inputbox "Enter Customer Name:" 8 40 3>&1 1>&2 2>&3)
  phone=$(dialog --inputbox "Enter Phone Number:" 8 40 3>&1 1>&2 2>&3)
  email=$(dialog --inputbox "Enter Email:" 8 40 3>&1 1>&2 2>&3)
  echo "$name,$phone,$email" >> customers.txt
  dialog --msgbox "Customer added successfully!" 6 20
}

# Function: View all customers
view_customers() {
  customer_list=$(awk -F"," '{printf "Name: %s, Phone: %s, Email: %s\n", $1, $2, $3}' customers.txt)
  dialog --title "Customer List" --msgbox "$customer_list" 20 70
}

# Function: Rent a car
rent_car() {
  car_id=$(dialog --inputbox "Enter Car ID:" 8 40 3>&1 1>&2 2>&3)
  rental_date=$(dialog --inputbox "Enter Rental Date (YYYY-MM-DD):" 8 40 3>&1 1>&2 2>&3)
  return_date=$(dialog --inputbox "Enter Return Date (YYYY-MM-DD):" 8 40 3>&1 1>&2 2>&3)

  if [ -z "$car_id" ] || [ -z "$rental_date" ] || [ -z "$return_date" ]; then
    dialog --msgbox "All fields are required!" 6 30
    return
  fi

  if ! date -d "$rental_date" &>/dev/null || ! date -d "$return_date" &>/dev/null; then
    dialog --msgbox "Invalid date format! Please use YYYY-MM-DD." 6 50
    return
  fi


  if [ "$(date -d "$return_date" +%s)" -lt "$(date -d "$rental_date" +%s)" ]; then
    dialog --msgbox "Return date cannot be earlier than rental date!" 6 50
    return
  fi


  rented=$(awk -F"," -v id="$car_id" '$1 == id && $5 == "Ongoing" {print "Rented"}' rentals.txt)

  if [ -z "$rented" ]; then

    days=$(( ($(date -d "$return_date" +%s) - $(date -d "$rental_date" +%s)) / 86400 ))

    rate=$(awk -F"," -v id="$car_id" 'NR == id {print $4}' cars.txt)
    total=$(echo "$days * $rate" | bc)


    echo "$car_id,$rental_date,$return_date,$total,Ongoing" >> rentals.txt


    dialog --msgbox "Car rented successfully! Total cost: $total" 6 40
  else
    dialog --msgbox "Car is already rented!" 6 30
  fi
}

# Function: Return a car
return_car() {

  car_id=$(dialog --inputbox "Enter Car ID:" 8 40 3>&1 1>&2 2>&3)


  if [ -z "$car_id" ]; then
    dialog --msgbox "Car ID cannot be empty!" 6 30
    return
  fi


  rental_info=$(awk -F"," -v id="$car_id" '$1 == id && $5 == "Ongoing" {print; exit}' rentals.txt)
  if [ -z "$rental_info" ]; then
    dialog --msgbox "Car is not currently rented or invalid car ID!" 6 30
    return
  fi


  rental_id=$(echo "$rental_info" | cut -d"," -f1)
  rental_date=$(echo "$rental_info" | cut -d"," -f2)
  expected_return_date=$(echo "$rental_info" | cut -d"," -f3)
  rate=$(echo "$rental_info" | cut -d"," -f4)


  actual_return_date=$(dialog --inputbox "Enter Actual Return Date (YYYY-MM-DD):" 8 40 3>&1 1>&2 2>&3)


  if ! date -d "$actual_return_date" &>/dev/null; then
    dialog --msgbox "Invalid date format! Please use YYYY-MM-DD." 6 50
    return
  fi


  overdue_days=$(( ( $(date -d "$actual_return_date" +%s) - $(date -d "$expected_return_date" +%s) ) / 86400 ))


  if [ "$overdue_days" -gt 0 ]; then
    fine=$(( overdue_days * 50 ))  # ₹50 per day fine
    dialog --msgbox "Car returned late by $overdue_days days. Fine: ₹$fine" 6 40
  else
    fine=0
    dialog --msgbox "Car returned on time. No fine." 6 30
  fi


  temp_file=$(mktemp)
  awk -F"," -v id="$rental_id" -v fine="$fine" -v OFS="," '$1 == id {$5 = "Completed"; $6 = fine} 1' rentals.txt > "$temp_file" && mv "$temp_file" rentals.txt


  dialog --msgbox "Car returned successfully! Total fine: ₹$fine" 6 40
}

# Function: Rental log
rental_log() {

  rental_list=$(awk -F"," '{printf "Rental ID: %d, Car ID: %s, Rental Date: %s, Return Date: %s, Total Cost: %s\n", NR, $1, $2, $3, $4, $5}' rentals.txt)
  dialog --title "Rental Log" --msgbox "$rental_list" 20 70
}

# Main menu
while true; do
  choice=$(dialog --menu "Car Rental System" 15 50 8 \
    1 "Add a Car" \
    2 "View Cars" \
    3 "Add a Customer" \
    4 "View Customers" \
    5 "Rent a Car" \
    6 "Return a Car" \
    7 "Rental Log" \
    8 "Exit" 2>&1 >/dev/tty)

  case $choice in
    1) add_car ;;
    2) view_cars ;;
    3) add_customer ;;
    4) view_customers ;;
    5) rent_car ;;
    6) return_car ;;
    7) rental_log ;;
    8) break ;;
    *) dialog --msgbox "Invalid choice! Please try again." 6 20 ;;
  esac
done

