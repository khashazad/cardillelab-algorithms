# Use a base image with Windows
FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Set the working directory
WORKDIR /pest

# Copy the PEST software into the container
COPY ./pest.exe /pest/pest.exe

# Set the path to include the PEST directory
ENV PATH="C:\\pest;%PATH%"
