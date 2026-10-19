from flex_travel_allocation.capacity_analysis import train_filling_by_hour


def test_sales_potential_scales_each_product_by_its_utilisation(spark):
    travels = spark.createDataFrame(
        [
            (8, "Flex Abonnement 120", 1),
            (8, "Flex Abonnement 120", 1),
            (8, "Flex Abonnement 6", 1),
            (8, "Flex Abonnement 6", 1),
            (8, "Flex Abonnement 6", 1),
            (17, "Flex Abonnement 120", 1),
        ],
        "travel_hour int, article_description_nl string, pax int",
    )
    utilisation = spark.createDataFrame(
        [("Flex Abonnement 120", 0.5), ("Flex Abonnement 6", 1.0)],
        "article_description_nl string, utilisation double",
    )

    rows = {
        row["travel_hour"]: row
        for row in train_filling_by_hour(travels, utilisation).collect()
    }

    # hour 8: 2 pax at 50% util -> 4 potential, plus 3 pax at 100% -> 3 => 5 actual, 7 potential
    assert rows[8]["allocation_pax"] == 5
    assert rows[8]["sales_potential_pax"] == 7
    # hour 17: 1 pax at 50% util -> 2 potential
    assert rows[17]["allocation_pax"] == 1
    assert rows[17]["sales_potential_pax"] == 2
