<template>
  <div class="container">
    <div class="video-container">
      <div class="video-stream">
        <h2>Original Feed</h2>
        <img :src="videoFeedUrl" alt="Original Feed" width="100%">
      </div>
    </div>
    <button type="submit" @click="isReporting ? stop_report() : start_report()">
      {{ isReporting ? 'Save Report' : 'Start Report' }}
    </button>
    <div>

      <div class="report-container">
        <h1>Reports</h1>
        <div class="report" v-for="report in reports" :key="report.name">
          <h2>{{ report.name }}</h2>
          <p>Date: {{ formatDateTime(report.date) }}</p>
          <table>
            <thead>
              <tr>
                <th>Hairline</th>
                <th>Settlement</th>
                <th>Structural</th>
                <th>Annotated Image</th>
                <th>Original Image</th>
                <th>DateTime</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in report.report" :key="item.datetime">
                <td>{{ item.Hairline }}</td>
                <td>{{ item.Settlement }}</td>
                <td>{{ item.Structural }}</td>
                <td><img class="report_image" :src="bas64image(item.annotated_image)" alt="Annotated" /></td>
                <td><img class="report_image" :src="bas64image(item.image)" alt="Original" /></td>
                <td>{{ formatDateTime(item.datetime) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      videoFeedUrl: 'http://127.0.0.1:8000/video_feed',
      isReporting: false,
      reports: []
    };
  },
  mounted: async function () {
    await this.get_reports()
    console.log(this.reports)
  },
  methods: {
    formatDateTime(dateInput) {
      const date = new Date(dateInput);

      if (isNaN(date)) {
        return 'Invalid Date';
      }

      const options = {
        weekday: 'long',  // "Monday"
        year: 'numeric',  // "2024"
        month: 'long',    // "May"
        day: 'numeric',   // "22"
        hour: '2-digit',  // "02 PM"
        minute: '2-digit',// "30"
        second: '2-digit' // "15"
      };

      return date.toLocaleDateString('en-US', options);
    },
    bas64image(image) {
      return `data:image/png;base64,${image}`
    },
    async get_reports() {
      const data = await axios.get('http://127.0.0.1:8000/get_reports');
      this.reports = data.data.data
    },
    async start_report() {
      try {
        const data = await axios.post('http://127.0.0.1:8000/start_report');
        console.log(data)
        this.isReporting = true
      } catch (error) {
        console.error('Error reporting', error);
      }
    },
    async stop_report() {
      try {
        const data = await axios.post('http://127.0.0.1:8000/stop_report');
        console.log(data)
        this.isReporting = false
        this.get_reports()
      } catch (error) {
        console.error('Error stopping report', error);
      }
    },
    updateFeed() {
      setInterval(() => {
        // Append a timestamp to force the browser to refresh the image
        this.$refs.videoFeed.src = this.videoFeedUrl + '?' + new Date().getTime();
      }, 100);
    }
  }
};
</script>

<style>
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f2f2f2;
}

.container {
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
}

.video-container {
  display: flex;
  justify-content: space-around;
  width: 100%;
  background-color: white;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
  padding: 20px;
  border-radius: 10px;
}

.video-stream {
  border: 1px solid #ddd;
  border-radius: 5px;
  overflow: hidden;
}

.video-stream img {
  width: 100%;
  height: 500px;
}

h1,
h2 {
  margin: 0;
  padding: 0;
  text-align: center;
}

h1 {
  font-size: 2.5em;
  color: #333;
  margin-bottom: 30px;
}

h2 {
  font-size: 1.5em;
  color: #555;
  margin-bottom: 20px;
}

form {
  margin-top: 30px;
  display: flex;
  justify-content: center;
}

button {
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 5px;
}

button:hover {
  background-color: #45a049;
}




.report-container {
  max-width: 800px;
  margin: auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 20px;
}

th,
td {
  border: 1px solid #ccc;
  padding: 10px;
  text-align: center;
}

th {
  background-color: #f4f4f4;
}

.report_image {
  max-width: 200px;
  height: auto;
}
</style>
