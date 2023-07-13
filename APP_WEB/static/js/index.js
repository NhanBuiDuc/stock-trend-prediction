index = {
  build: function (symbol) {
    let object = this;
    //object.buildSectionNeutral();
    if (object.currentData) {
      object.buildSectionNeutral(object.currentData, symbol); // Use the current data if available
      console.log("Building Technical Analysis");
    } else {
      console.log("No data available.");
    }
    object.scrollEvent();
  },

  getData: async function (filename) {
    const path = "../static/file/";
    let filepath = path + filename;
    try {
      const response = await fetch(filepath);
      if (response) {
        const data = await response.json();
        if (data) return data;
      }
    } catch (error) {
      console.error("Error:", error);
    }
  },

  scrollEvent: function () {
    let footer = document.querySelector("footer");
    let mainfuncbuttons = document.querySelector(".main-func-buttons");

    window.addEventListener("scroll", function (e) {
      let screen = window.scrollY + window.innerHeight;
      let targetPosition = footer.offsetTop;

      if (screen + 100 > targetPosition) {
        mainfuncbuttons.style.bottom = "5rem";
      } else {
        mainfuncbuttons.style.bottom = "2rem";
      }
    });
  },

  buildSectionNeutral: function (data, symbol) {
    let object = this;

    var table1 = document.getElementById("table1");
    var tbody1 = table1.querySelector("tbody");

    var rows1 = tbody1.querySelectorAll("tr");
    for (var i = rows1.length - 1; i > 0; i--) {
      tbody1.removeChild(rows1[i]);
    }

    var table2 = document.getElementById("table2");
    var tbody2 = table2.querySelector("tbody");

    var rows2 = tbody2.querySelectorAll("tr");
    for (var i = rows2.length - 1; i > 0; i--) {
      tbody2.removeChild(rows2[i]);
    }

    // jquery code
    $(document).ready(function () {
      $.getJSON(`../static/file/${symbol}_signal.json`, function (data) {
        data = data.reverse();

        let sel = document.getElementById("sel");
        sel.innerHTML = ``;

        var optionElement = $("<option>");
        optionElement.attr("disabled", true);
        optionElement.attr("selected", true);
        optionElement.attr("value", "");
        optionElement.text("Select date");

        // Append the new option element to the select element
        $("#sel").append(optionElement);
        $.each(data, function (i, option) {
          $("#sel").append($("<option/>").text(option.date));
        });
      });
    });

    // vanilla js code
    // object.getData("../static/file/AAPL_signal.json")
    //     .then(function (data) {
    //         let sel = document.getElementById("sel");
    //         data = data.reverse();
    //         data.forEach(childData => {
    //             let option = document.createElement("option");
    //             option.innerText = childData.date;
    //             sel.append(option);
    //         });
    //     })
    //     .catch(function (error) {
    //         console.error('Error:', error);
    //     });

    $("#sel").change(function () {
      var table = document.getElementById("table1");
      var tbody = table.getElementsByTagName("tbody")[0];

      var e = document.getElementById("sel");
      result = e.options[e.selectedIndex].value;
      var selltxt2 = document.getElementById("sell2");
      var neutraltxt2 = document.getElementById("neutral2");
      var buytxt2 = document.getElementById("buy2");
      var total2 = document.getElementById("total2");
      var arrow2 = document.getElementById("arr2");
      var selltxt1 = document.getElementById("sell1");
      var neutraltxt1 = document.getElementById("neutral1");
      var buytxt1 = document.getElementById("buy1");
      var total1 = document.getElementById("total1");
      var arrow1 = document.getElementById("arr1");
      var selltxt3 = document.getElementById("sell3");
      var neutraltxt3 = document.getElementById("neutral3");
      var buytxt3 = document.getElementById("buy3");
      var total3 = document.getElementById("total3");
      var arrow3 = document.getElementById("arr3");
      $.getJSON(`../static/file/${symbol}_signal.json`, function (data) {
        var sell2 = 0;
        var neutral2 = 0;
        var buy2 = 0;
        var sell1 = 0;
        var neutral1 = 0;
        var buy1 = 0;

        for (var i = 0; i < data.length; i++) {
          if (data[i].date == result) {
            var table1 = document.getElementById("table1");

            table1.rows[1].cells[1].innerHTML = data[i].RSI_14.toFixed(2);
            if (data[i].s_RSI > 0) {
              table1.rows[1].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_RSI == 0) {
              table1.rows[1].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_RSI < 0) {
              table1.rows[1].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[2].cells[1].innerHTML = data[i].MACD_12_26_9.toFixed(2);
            if (data[i].s_MACD > 0) {
              table1.rows[2].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_MACD == 0) {
              table1.rows[2].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_MACD < 0) {
              table1.rows[2].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[3].cells[1].innerHTML = data[i].STOCHRSIk_14_14_3_3.toFixed(2);
            if (data[i].s_STOCHRSI > 0) {
              table1.rows[3].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_STOCHRSI == 0) {
              table1.rows[3].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_STOCHRSI < 0) {
              table1.rows[3].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[4].cells[1].innerHTML = data[i].WILLR_14.toFixed(2);
            if (data[i].s_WILLR > 0) {
              table1.rows[4].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_WILLR == 0) {
              table1.rows[4].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_WILLR < 0) {
              table1.rows[4].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[5].cells[1].innerHTML = data[i].MOM_10.toFixed(2);
            if (data[i].s_MOM > 0) {
              table1.rows[5].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_MOM == 0) {
              table1.rows[5].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_MOM < 0) {
              table1.rows[5].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[6].cells[1].innerHTML = data[i]["CCI_20_0.015"].toFixed(2);
            if (data[i].s_CCI > 0) {
              table1.rows[6].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_CCI == 0) {
              table1.rows[6].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_CCI < 0) {
              table1.rows[6].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }
            table1.rows[7].cells[1].innerHTML = data[i].UO_7_14_28.toFixed(2);
            if (data[i].s_UO > 0) {
              table1.rows[7].cells[2].innerHTML = "Buy";
              buy1 = parseInt(buy1) + 1;
            } else if (data[i].s_UO == 0) {
              table1.rows[7].cells[2].innerHTML = "Neutral";
              neutral1 = parseInt(neutral1) + 1;
            } else if (data[i].s_UO < 0) {
              table1.rows[7].cells[2].innerHTML = "Sell";
              sell1 = parseInt(sell1) + 1;
            }

            var table2 = document.getElementById("table2");
            table2.rows[1].cells[1].innerHTML = data[i].SMA_10.toFixed(2);
            if (data[i].s_SMA_10 > 0) {
              table2.rows[1].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_10 == 0) {
              table2.rows[1].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_10 < 0) {
              table2.rows[1].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[2].cells[1].innerHTML = data[i].EMA_10.toFixed(2);
            if (data[i].s_EMA_10 > 0) {
              table2.rows[2].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_10 == 0) {
              table2.rows[2].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_10 < 0) {
              table2.rows[2].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[3].cells[1].innerHTML = data[i].SMA_20.toFixed(2);
            if (data[i].s_SMA_20 > 0) {
              table2.rows[3].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_20 == 0) {
              table2.rows[3].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_20 < 0) {
              table2.rows[3].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[4].cells[1].innerHTML = data[i].EMA_20.toFixed(2);
            if (data[i].s_EMA_20 > 0) {
              table2.rows[4].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_20 == 0) {
              table2.rows[4].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_20 < 0) {
              table2.rows[4].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[5].cells[1].innerHTML = data[i].SMA_30.toFixed(2);
            if (data[i].s_SMA_30 > 0) {
              table2.rows[5].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_30 == 0) {
              table2.rows[5].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_30 < 0) {
              table2.rows[5].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[6].cells[1].innerHTML = data[i].EMA_30.toFixed(2);
            if (data[i].s_EMA_30 > 0) {
              table2.rows[6].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_30 == 0) {
              table2.rows[6].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_30 < 0) {
              table2.rows[6].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[7].cells[1].innerHTML = data[i].SMA_50.toFixed(2);
            if (data[i].s_SMA_50 > 0) {
              table2.rows[7].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_50 == 0) {
              table2.rows[7].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_50 < 0) {
              table2.rows[7].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[8].cells[1].innerHTML = data[i].EMA_50.toFixed(2);
            if (data[i].s_EMA_50 > 0) {
              table2.rows[8].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_50 == 0) {
              table2.rows[8].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_50 < 0) {
              table2.rows[8].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[9].cells[1].innerHTML = data[i].SMA_100.toFixed(2);
            if (data[i].s_SMA_100 > 0) {
              table2.rows[9].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_100 == 0) {
              table2.rows[9].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_100 < 0) {
              table2.rows[9].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[10].cells[1].innerHTML = data[i].EMA_100.toFixed(2);
            if (data[i].s_EMA_100 > 0) {
              table2.rows[10].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_100 == 0) {
              table2.rows[10].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_100 < 0) {
              table2.rows[10].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[11].cells[1].innerHTML = data[i].SMA_200.toFixed(2);
            if (data[i].s_SMA_200 > 0) {
              table2.rows[11].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_SMA_200 == 0) {
              table2.rows[11].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_SMA_200 < 0) {
              table2.rows[11].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[12].cells[1].innerHTML = data[i].EMA_200.toFixed(2);
            if (data[i].s_EMA_200 > 0) {
              table2.rows[12].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_EMA_200 == 0) {
              table2.rows[12].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_EMA_200 < 0) {
              table2.rows[12].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
            table2.rows[13].cells[1].innerHTML = data[i].HMA_9.toFixed(2);
            if (data[i].s_HMA > 0) {
              table2.rows[13].cells[2].innerHTML = "Buy";
              buy2 = parseInt(buy2) + 1;
            } else if (data[i].s_HMA == 0) {
              table2.rows[13].cells[2].innerHTML = "Neutral";
              neutral2 = parseInt(neutral2) + 1;
            } else if (data[i].s_HMA < 0) {
              table2.rows[13].cells[2].innerHTML = "Sell";
              sell2 = parseInt(sell2) + 1;
            }
          }
        }
        buytxt1.innerHTML = buy1;
        neutraltxt1.innerHTML = neutral1;
        selltxt1.innerHTML = sell1;
        if (buy1 > neutral1) {
          if (buy1 > sell1) {
            total.innerHTML = "Buy";
            arrow1.className = "arrow-container speed-150";
          }
        } else if (neutral1 > sell1) {
          total1.innerHTML = "Neutral";
          arrow1.className = "arrow-container speed-90";
        } else {
          total1.innerHTML = "Sell";
          arrow1.className = "arrow-container speed-30";
        }

        buytxt2.innerHTML = buy2;
        neutraltxt2.innerHTML = neutral2;
        selltxt2.innerHTML = sell2;
        if (buy2 > neutral2) {
          if (buy2 > sell2) {
            total2.innerHTML = "Buy";
            arrow2.className = "arrow-container speed-150";
          }
        } else if (neutral2 > sell2) {
          total2.innerHTML = "Neutral";
          arrow2.className = "arrow-container speed-90";
        } else {
          total2.innerHTML = "Sell";
          arrow2.className = "arrow-container speed-30";
        }

        buytxt3.innerHTML = buy1 + buy2;
        neutraltxt3.innerHTML = neutral1 + neutral2;
        selltxt3.innerHTML = sell1 + sell2;
        if (buy1 + buy2 > neutral1 + neutral2) {
          if (buy2 > sell2) {
            total3.innerHTML = "Buy";
            arrow3.className = "arrow-container speed-150";
          }
        } else if (neutral1 + neutral2 > sell1 + sell2) {
          total3.innerHTML = "Neutral";
          arrow3.className = "arrow-container speed-90";
        } else {
          total3.innerHTML = "Sell";
          arrow3.className = "arrow-container speed-30";
        }
      });
    });

    $(document).ready(function () {
      $.getJSON(`../static/file/${symbol}_signal.json`, function (data) {
        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(10)</td>";

        row += "<td>" + data[0].SMA_10 + "</td>";

        if (data[0].s_SMA_10 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_10 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_10 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(10)</td>";

        row += "<td>" + data[0].EMA_10 + "</td>";

        if (data[0].s_EMA_10 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_10 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_10 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";
        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(20)</td>";

        row += "<td>" + data[0].SMA_20 + "</td>";

        if (data[0].s_SMA_20 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_20 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_20 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(20)</td>";

        row += "<td>" + data[0].EMA_20 + "</td>";

        if (data[0].s_EMA_20 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_20 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_20 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(30)</td>";

        row += "<td>" + data[0].SMA_30 + "</td>";

        if (data[0].s_SMA_30 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_30 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_30 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(30)</td>";

        row += "<td>" + data[0].EMA_30 + "</td>";

        if (data[0].s_EMA_30 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_30 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_30 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(50)</td>";

        row += "<td>" + data[0].SMA_50 + "</td>";

        if (data[0].s_SMA_50 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_50 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_50 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(50)</td>";

        row += "<td>" + data[0].EMA_50 + "</td>";

        if (data[0].s_EMA_50 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_50 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_50 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(100)</td>";

        row += "<td>" + data[0].SMA_100 + "</td>";

        if (data[0].s_SMA_100 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_100 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_100 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(100)</td>";

        row += "<td>" + data[0].EMA_100 + "</td>";

        if (data[0].s_EMA_100 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_100 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_100 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Simple Moving Average(200)</td>";

        row += "<td>" + data[0].SMA_200 + "</td>";

        if (data[0].s_SMA_200 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_SMA_200 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_SMA_200 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Exponential Moving Average(200)</td>";

        row += "<td>" + data[0].EMA_200 + "</td>";

        if (data[0].s_EMA_200 > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_EMA_200 == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_EMA_200 < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Hull Moving Average(9)</td>";

        row += "<td>" + data[0].HMA_9 + "</td>";

        if (data[0].s_HMA > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_HMA == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_HMA < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table2").append(row);
      });
    });

    $(document).ready(function () {
      $.getJSON(`../static/file/${symbol}_signal.json`, function (data) {
        var row = "";

        row += "<tr>";

        row += "<td>Relative Strength Index(14)</td>";

        row += "<td>" + data[0].RSI_14 + "</td>";

        if (data[0].s_RSI > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_RSI == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_RSI < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>MACD Level(12, 26)</td>";

        row += "<td>" + data[0].MACD_12_26_9 + "</td>";

        if (data[0].s_MACD > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_MACD == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_MACD < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";
        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Stochastic %K(14, 3, 3)</td>";

        row += "<td>" + data[0].STOCHRSIk_14_14_3_3 + "</td>";

        if (data[0].s_STOCHRSI > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_STOCHRSI == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_STOCHRSI < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Williams Percent Range(14)</td>";

        row += "<td>" + data[0].WILLR_14 + "</td>";

        if (data[0].s_WILLR > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_WILLR == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_WILLR < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Momentum(10)</td>";

        row += "<td>" + data[0].MOM_10 + "</td>";

        if (data[0].s_MOM > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_MOM == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_MOM < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Commodity Channel Index(20)</td>";

        row += "<td>" + data[0]["CCI_20_0.015"] + "</td>";

        if (data[0].s_CCI > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_CCI == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_CCI < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);

        var row = "";

        row += "<tr>";

        row += "<td>Ultimate Oscillator(7, 14, 28)</td>";

        row += "<td>" + data[0].UO_7_14_28 + "</td>";

        if (data[0].s_UO > 0) {
          row += "<td>Buy</td>";
        } else if (data[0].s_UO == 0) {
          row += "<td>Neutral</td>";
        } else if (data[0].s_UO < 0) {
          row += "<td>Sell</td>";
        }
        row += "</tr>";

        $("#table1").append(row);
      });
    });
  },
};
