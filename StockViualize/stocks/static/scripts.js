document.addEventListener("DOMContentLoaded", function () {
    // 初始化 Flatpickr 日期選擇器
    flatpickr(".date-picker", {
        dateFormat: "Y-m-d",
        maxDate: "today", // 不允許選擇未來日期
        locale: "zh" // 使用中文本地化
    });

    // 校驗開始和結束日期
    const startDateInput = document.getElementById("start_date");
    const endDateInput = document.getElementById("end_date");

    endDateInput.addEventListener("change", function () {
        const startDate = new Date(startDateInput.value);
        const endDate = new Date(endDateInput.value);

        if (endDate < startDate) {
            alert("結束日期不能早於開始日期！");
            endDateInput.value = "";
        }
    });
});
