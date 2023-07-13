chart = {
  
  upElements: "<i class='fa-solid fa-arrow-up' style='color: #289125;'> UP</i>",
  downElements: "<i class='fa-solid fa-arrow-down' style='color: #c81e1e;'> DOWN</i>",

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

  build: function (symbol) {
    let object = this;
    console.log("Building chart...");
    if (object.currentData) {
      object.buildHighchart(object.currentData, symbol); // Use the current data if available

      // let stockTable = document.getElementById("stockTable");
      // let tbody = stockTable.querySelector("tbody");
      // tbody.innerHTML = ``;
      object.buildOnchangeOutputSize();
      object.buildOnchangedatePicker();
    } else {
      console.log("No data available.");
    }
  },

  buildOnchangeOutputSize: function () {
    let object = this;
    let outputsize = document.getElementById("outputsize");
    outputsize.onchange = function () {
      let symbol = document.getElementById("titleSignal").innerHTML;
      object.renderDataToTable(symbol);
    };
  },
  buildOnchangedatePicker: function () {
    let object = this;
    let datePicker = document.getElementById("datePicker");
    datePicker.onchange = function () {
      let symbol = document.getElementById("titleSignal").innerHTML;
      object.renderDataToTable(symbol);
    };
  },

  buildHighchart: function (data, symbol) {
    let object = this;

    // split the data set into ohlc and volume
    var ohlc = [],
      volume = [],
      formatData = [],
      dataLength = data.length;

    for (var i = 0; i < dataLength; i += 1) {
      ohlc.push([
        new Date(data[i]["date"].split("/").reverse().join("/")).getTime(), // the date
        data[i]["1. open"], // open
        data[i]["2. high"], // high
        data[i]["3. low"], // low
        data[i]["4. close"], // close
      ]);

      volume.push([
        new Date(data[i]["date"].split("/").reverse().join("/")).getTime(), // the date
        data[i]["6. volume"], // the volume
      ]);

      formatData.push([
        new Date(data[i]["date"].split("/").reverse().join("/")).getTime(), // the date
        Number(data[i]["1. open"]), // open
        Number(data[i]["2. high"]), // high
        Number(data[i]["3. low"]), // low
        Number(data[i]["4. close"]), // close
        data[i]["6. volume"], // the volume
      ]);
    }

    // create the chart
    Highcharts.stockChart(
      "candlechart",
      {
        chart: {
          height: 500,
        },
        title: {
          text: symbol + " Historical",
        },
        subtitle: {
          text: "All indicators",
        },
        accessibility: {
          series: {
            descriptionFormat: "{seriesDescription}.",
          },
          description: "Use the dropdown menus above to display different indicator series on the chart.",
          screenReaderSection: {
            beforeChartFormat:
              "<{headingTagName}>{chartTitle}</{headingTagName}><div>{typeDescription}</div><div>{chartSubtitle}</div><div>{chartLongdesc}</div>",
          },
        },
        legend: {
          enabled: true,
        },
        rangeSelector: {
          selected: 2,
        },
        yAxis: [
          {
            height: "60%",
          },
          {
            top: "60%",
            height: "20%",
          },
          {
            top: "80%",
            height: "20%",
          },
        ],
        plotOptions: {
          series: {
            showInLegend: true,
            accessibility: {
              exposeAsGroupOnly: true,
            },
          },
        },
        series: [
          {
            type: "candlestick",
            id: "aapl",
            name: symbol,
            data: formatData,
          },
          {
            type: "column",
            id: "volume",
            name: "Volume",
            data: volume,
            yAxis: 1,
          },
          {
            type: "pc",
            id: "overlay",
            linkedTo: "aapl",
            yAxis: 0,
          },
          {
            type: "macd",
            id: "oscillator",
            linkedTo: "aapl",
            yAxis: 2,
          },
        ],
      },
      function (chart) {
        document.getElementById("overlays").addEventListener("change", function (e) {
          var series = chart.get("overlay");

          if (series) {
            series.remove(false);
            chart.addSeries({
              type: e.target.value,
              linkedTo: "aapl",
              id: "overlay",
            });
          }
        });

        document.getElementById("oscillators").addEventListener("change", function (e) {
          var series = chart.get("oscillator");

          if (series) {
            series.remove(false);
            chart.addSeries({
              type: e.target.value,
              linkedTo: "aapl",
              id: "oscillator",
              yAxis: 2,
            });
          }
        });
      }
    );
  },

  formatDataSet: function (dataset) {
    let json = {
      Total: {},
      BUY: {},
      NEU: {},
      SELL: {},
    };
    dataset.forEach((data) => {
      let year = data.date.split("/")[2];

      json["Total"][`${year}`] = `${data["BUY"] + data["SELL"] + data["NEU"]}`;
      json["BUY"][`${year}`] = `${data["BUY"]}`;
      json["NEU"][`${year}`] = `${data["NEU"]}`;
      json["SELL"][`${year}`] = `${data["SELL"]}`;
    });

    return json;
  },

  renderDataToTable: function (value) {
    let object = this;

    let actualPrice = document.getElementById("actualPrice");
    let currentPrice = document.getElementById("currentPrice");
    let datePicker = document.getElementById("datePicker");
    let stockTable = document.getElementById("stockTable");
    let tbody = stockTable.querySelector("tbody");
    // tbody.innerHTML = ``;
    object.getDataTable(value, function (res) {
      if (res) {
        let outputsize = document.getElementById("outputsize").value;
        let dateValue = document.getElementById("datePicker").value;
        // Split the retrieved value using the hyphen "-"
        const [year, month, day] = dateValue.split("-");

        console.log("data: ", res);

        // Rearrange the split values to the desired format ("01-07-2023")
        const formattedDate = `${year}-${month}-${day}`;
        let datePicker = formattedDate;

        currentPrice.innerText = (res[datePicker] ? res[datePicker]["current"] : "0");
        actualPrice.innerText = (res[datePicker] ? res[datePicker][outputsize]["actual"]: "0");
        

        let tr = document.createElement("tr");
        tr.innerHTML = `
                <td>${datePicker}</td>
                <td>${value}</td>
                <td>${res[datePicker][outputsize]["lstm"] == "UP" ? object.upElements : object.downElements}</td>
                <td>${res[datePicker][outputsize]["svm"] == "UP" ? object.upElements : object.downElements}</td>
                <td>${res[datePicker][outputsize]["xgboost"] == "UP" ? object.upElements : object.downElements}</td>
                <td>${res[datePicker][outputsize]["random forest"]  == "UP" ? object.upElements : object.downElements}</td>
                <td>${res[datePicker][outputsize]["ensembler"] == "UP" ? object.upElements : object.downElements}</td>`;
        
        if (tbody.childElementCount >= 4) {
          // Remove the last row from the table
          tbody.removeChild(tbody.lastChild);
        }
        tbody.insertBefore(tr, tbody.firstChild);
      }
    });
  },

  callApiGetPrice: function (value, callback) {
    let url = `http://127.0.0.1:5000/execute/${value}`;
    fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => {
        console.log("respond: ", response);
        return response.json();
      })
      .then((data) => {
        console.log("Respone API", data);
        callback(data);
      })
      .catch((err) => {
        console.log("API went wrong.", err);
      });
  },



  getDataTable: async function (value, callback) {
    let object = this;

    object
      .getData("prediction.json")
      .then(function (data) {
        let dataSymbol = data[`${value}`];
        callback(dataSymbol);
      })
      .catch(function (error) {
        console.error("Error:", error);
      });
  },

  // getDataTable: function(value, callback) {
  //     let object = this;
  //     const oob = getValue();
  //     console.log(oob);

  //     const list = ['svm'];

  //     const payload = {
  //         symbol: value,
  //         window_size: oob.windowsize,
  //         output_size: oob.outputsize,
  //         model_type_list: list
  //     };
  //     console.log('pay_load: ', JSON.stringify(payload))
  //     const url = '/execute';
  //     fetch(url, {
  //         method: 'POST',
  //         headers: {
  //             'Content-Type': 'application/json'
  //         },
  //         body: JSON.stringify(payload)

  //     })
  //         .then(response => {
  //             console.log('Response', response);
  //             return response.json();
  //         })
  //         .then(data => {
  //             console.log('API Response', data);
  //             callback(data);
  //         })
  //         .catch(err => {
  //             console.log('API Error', err);
  //         });
  // },
};
